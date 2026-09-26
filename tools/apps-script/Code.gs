/**
 * Project Tengen — Sheets export endpoint.
 *
 * Paste this into a spreadsheet's Apps Script editor (Extensions > Apps Script) and deploy it as
 * a Web App. It runs as the sheet's owner inside Workspace, so it needs no Google Cloud project
 * and no service account — which is the whole point, since school accounts usually have Cloud
 * switched off.
 *
 * Everything lands in ONE tab, named "Tengen". If the spreadsheet already has one (any
 * capitalisation), that tab is used; otherwise it is created. Inside it, each kind of export is
 * its own block, side by side:
 *
 *   row 1   Tengen aggregates        (title — how the block is found again)
 *   row 2   row_key | match_id | team | ...   (headers)
 *   row 3+  one row per team per match
 *
 * Raw-event exports get a second block to the right, titled "Tengen raw events". A new block is
 * always placed after the last used column, so anything the team already keeps in the tab is
 * never overwritten.
 *
 * Setup, once:
 *   1. Open the spreadsheet > Extensions > Apps Script. Replace everything with this file.
 *   2. Edit SECRET below to a long random string. Keep it; the ingest service needs the same one.
 *   3. Deploy > New deployment > type "Web app".
 *        Execute as:      Me
 *        Who has access:  Anyone
 *      "Anyone" is required — the ingest service is not signed in as a Google user. That is why
 *      the secret exists: the URL alone must not be enough to write to your sheet.
 *   4. Copy the /exec URL. Put both values in ingest/.env, which is git-ignored:
 *        APPS_SCRIPT_URL=https://script.google.com/macros/s/..../exec
 *        APPS_SCRIPT_SECRET=the same string as SECRET below
 *
 * Re-deploy after any edit: Apps Script serves the last *deployed* version, not the last saved
 * one, so an edited-but-undeployed script keeps running the old code and looks like nothing
 * changed. Deploy > Manage deployments > edit > New version keeps the same URL.
 */

var SECRET = 'CHANGE-ME-to-a-long-random-string';

var TAB_NAME = 'Tengen';

// The ingest service says which kind of rows it is sending; each kind is one block.
var BLOCK_TITLES = {
  aggregates: 'Tengen aggregates',
  raw_events: 'Tengen raw events'
};

var TITLE_ROW = 1;
var HEADER_ROW = 2;
var FIRST_DATA_ROW = 3;

function doPost(e) {
  try {
    var body = JSON.parse(e.postData.contents);

    if (!SECRET || SECRET === 'CHANGE-ME-to-a-long-random-string') {
      return json({ ok: false, error: 'the script SECRET has not been set' });
    }
    if (body.secret !== SECRET) {
      // Deliberately vague: a precise message would help someone guess.
      return json({ ok: false, error: 'rejected' });
    }

    var kind = String(body.tab || 'aggregates');
    var title = BLOCK_TITLES[kind] || ('Tengen ' + kind);
    var headers = body.headers || [];
    var rows = body.rows || [];
    var width = headers.length;

    var book = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = findOrCreateTab(book);

    // Find this block by its title in row 1, or start it two columns past everything already in
    // the tab — one blank column between blocks keeps them readable and apart.
    var start = findBlock(sheet, title);
    if (!start) {
      start = sheet.getLastColumn() === 0 ? 1 : sheet.getLastColumn() + 2;
      sheet.getRange(TITLE_ROW, start).setValue(title).setFontWeight('bold');
      if (sheet.getFrozenRows() < HEADER_ROW) sheet.setFrozenRows(HEADER_ROW);
    }

    // If the columns ever grow, the block must not spill into whatever sits to its right.
    var next = nextUsedColumn(sheet, start);
    if (next && start + width > next) {
      return json({ ok: false, error: 'the "' + title + '" block in the ' + TAB_NAME +
                   ' tab needs ' + width + ' columns but runs into column ' + next +
                   '; move or delete what is there' });
    }

    // Headers are rewritten every time, so they stay in step if the columns change.
    sheet.getRange(HEADER_ROW, start, 1, width).setValues([headers]).setFontWeight('bold');

    // The block's first column is the stable key. Read the block once and index it, so a
    // re-export REPLACES a row instead of appending a duplicate — the same idempotence guarantee
    // the Cloud path gives.
    var existing = [];
    if (sheet.getLastRow() >= FIRST_DATA_ROW) {
      existing = sheet.getRange(FIRST_DATA_ROW, start,
                                sheet.getLastRow() - FIRST_DATA_ROW + 1, width).getValues();
    }
    var keyToIndex = Object.create(null);
    var used = 0;   // rows of this block in use; the other block may be longer
    for (var i = 0; i < existing.length; i++) {
      var key = String(existing[i][0] || '');
      if (key) {
        keyToIndex[key] = i;
        used = i + 1;
      }
    }

    var written = 0;
    var skipped = 0;
    var appended = [];

    for (var r = 0; r < rows.length; r++) {
      var row = rows[r];
      var k = String(row[0]);
      if (k in keyToIndex) {
        var at = keyToIndex[k];
        if (sameRow(existing[at], row, width)) {
          skipped++;                     // already present and identical
        } else {
          sheet.getRange(FIRST_DATA_ROW + at, start, 1, width).setValues([pad(row, width)]);
          written++;
        }
      } else {
        appended.push(pad(row, width));
        written++;
      }
    }

    // One write for everything new, rather than one per row. Per-row calls are what exhaust the
    // quota on a big export.
    if (appended.length) {
      sheet.getRange(FIRST_DATA_ROW + used, start, appended.length, width).setValues(appended);
    }

    return json({
      ok: true,
      rows_written: written,
      rows_skipped: skipped,
      spreadsheet_url: book.getUrl()
    });
  } catch (err) {
    return json({ ok: false, error: String(err) });
  }
}

/** A GET is useful for checking the deployment is live, and which version, without writing. */
function doGet() {
  return json({ ok: true, service: 'tengen-sheets-export', writes_to: TAB_NAME,
                note: 'POST rows to this URL' });
}

/** The tab named "Tengen", matched loosely so " tengen" or "TENGEN" is not duplicated. */
function findOrCreateTab(book) {
  var sheets = book.getSheets();
  for (var i = 0; i < sheets.length; i++) {
    if (sheets[i].getName().trim().toLowerCase() === TAB_NAME.toLowerCase()) return sheets[i];
  }
  return book.insertSheet(TAB_NAME);
}

/** The column where the block titled `title` starts, or 0 if it is not in the tab yet. */
function findBlock(sheet, title) {
  var last = sheet.getLastColumn();
  if (last === 0) return 0;
  var cells = sheet.getRange(TITLE_ROW, 1, 1, last).getValues()[0];
  for (var c = 0; c < cells.length; c++) {
    if (String(cells[c]).trim().toLowerCase() === title.toLowerCase()) return c + 1;
  }
  return 0;
}

/** The first column right of `start` with anything in row 1 — the next block, if any. */
function nextUsedColumn(sheet, start) {
  var last = sheet.getLastColumn();
  if (last <= start) return 0;
  var cells = sheet.getRange(TITLE_ROW, start + 1, 1, last - start).getValues()[0];
  for (var c = 0; c < cells.length; c++) {
    if (String(cells[c]) !== '') return start + 1 + c;
  }
  return 0;
}

function pad(row, width) {
  var out = [];
  for (var i = 0; i < width; i++) out.push(i < row.length && row[i] != null ? row[i] : '');
  return out;
}

function sameRow(a, b, width) {
  for (var i = 0; i < width; i++) {
    var left = i < a.length && a[i] != null ? String(a[i]) : '';
    var right = i < b.length && b[i] != null ? String(b[i]) : '';
    if (left !== right) return false;
  }
  return true;
}

function json(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
                       .setMimeType(ContentService.MimeType.JSON);
}
