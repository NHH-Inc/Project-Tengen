# Sheets export without Google Cloud

Use this when the account holding the spreadsheet cannot create a Google Cloud project. That is
the normal state of a school Workspace account — an administrator has turned Cloud off, and no
change to this codebase can turn it back on.

An Apps Script bound to the spreadsheet runs **as the sheet's owner, inside Workspace**. No Cloud
project, no service account, no credential file. It publishes a URL, and the ingest service posts
rows to it.

Same rows, same stable keys, same idempotence as the service-account path. Which transport runs
is a question about account access, not behaviour.

## Setup — about five minutes, once

**1. Open the script editor.** In the spreadsheet: **Extensions → Apps Script**. Delete whatever
is there and paste all of `Code.gs`.

**2. Set the secret.** Near the top:

```js
var SECRET = 'CHANGE-ME-to-a-long-random-string';
```

Replace it with something long and random. This is the only thing standing between the URL and
anyone who finds it.

**3. Deploy.** **Deploy → New deployment → Web app**:

| setting | value |
|---|---|
| Execute as | **Me** |
| Who has access | **Anyone** |

"Anyone" is required and is not a mistake: the ingest service is a script, not a signed-in Google
user, so it cannot authenticate as one. That is exactly why the secret exists.

Google will ask you to authorise the script. It is your own script running on your own
spreadsheet.

**4. Copy the `/exec` URL** and put both values in `ingest/.env` (git-ignored — never commit
either):

```
APPS_SCRIPT_URL=https://script.google.com/macros/s/AKfy..../exec
APPS_SCRIPT_SECRET=the same string you set as SECRET
```

**5. Check it is live.** Open the `/exec` URL in a browser. You should see:

```json
{"ok":true,"service":"tengen-sheets-export","writes_to":"Tengen","note":"POST rows to this URL"}
```

No `writes_to` means the old version is still deployed — see "Editing without redeploying" below.

Then export as normal — `POST /api/export/sheets`. The response includes
`"transport": "apps_script"` so you can tell which path ran.

## Things that will catch you out

**Editing without redeploying.** Apps Script serves the last *deployed* version, not the last
saved one. Edit the script and the old code keeps running, which looks exactly like your change
having no effect. After any edit: **Deploy → Manage deployments → edit → New version**.

**`{"ok":false,"error":"rejected"}`** means the secret in `.env` and the one in the script do not
match. The message is deliberately vague — a precise one would help someone guessing.

**Everything goes in one tab named `Tengen`.** If the spreadsheet already has one (any
capitalisation), it is used; otherwise it is created. Renaming it will cause the next export to
create a fresh one. Earlier versions of this script wrote `aggregates` and `raw_events` tabs;
those are no longer written and can be deleted.

## Where the rows go

Each kind of export is its own block in the `Tengen` tab, side by side:

| row | aggregates block (usually columns A–N) | raw events block (to its right) |
|---|---|---|
| 1 | `Tengen aggregates` | `Tengen raw events` |
| 2 | headers: `row_key`, `match_id`, `team`, … | headers: `row_key`, `match_id`, `event_id`, … |
| 3+ | one row per team per match | one row per event |

The title in row 1 is how the script finds a block again, so leave it alone. A block is created
the first time that kind is exported, two columns past everything already in the tab — anything
the team keeps there is never overwritten. If the columns ever grow into the next block, the
export is refused with a message rather than overwriting it.

To use the numbers elsewhere in the spreadsheet, point formulas at the block, e.g. total shots
made by team 1234 across every exported match:

```
=SUMIFS(Tengen!G:G, Tengen!C:C, 1234)
```

## What it does with your data

Only writes. Each block's first column is a stable key, so re-exporting a match **replaces** its
rows rather than appending duplicates, and a row that is already present and identical is skipped
and counted in `rows_skipped`. Nothing is ever read back as a source of truth — per doc 3, Sheets
is an export destination, not storage.

The service-account transport (`ingest/sheets.py`) still writes separate `aggregates` and
`raw_events` tabs.
