# Hosting and storage

Written to be readable by anyone on the team, or by an AI assistant picking this up cold.

The question: **we don't want this running on someone's personal PC. Where does it go, and where
do the videos and images live?**

---

## 1. What actually has to run

The system is four jobs, and they have very different needs. Lumping them together is what makes
hosting look harder than it is.

| Job | Needs | Can it live in the cloud? |
|---|---|---|
| **Download video** (yt-dlp) | A *residential* internet connection | **No — see §2** |
| **Analyse video** (the C++ binary) | CPU, eventually a GPU | Expensive in the cloud |
| **Store the results** (database) | Almost nothing. Tiny. | **Yes, free** |
| **Serve the API + web app** | Almost nothing | **Yes, free or ~$4/mo** |

Only the first two are hard, and only the first one is hard for a reason people do not expect.

---

## 2. The constraint nobody expects: YouTube blocks servers

**yt-dlp downloads fail from datacenter IP addresses.** Not sometimes — routinely. You get
*"Sign in to confirm you're not a bot"*, or throttling to unusable speeds. YouTube treats traffic
from AWS, Oracle, Hetzner, DigitalOcean and every other host as suspicious, because that is where
scraping comes from.

This is the single biggest thing shaping the answer. It means **the downloader wants to run on a
home internet connection**, even when everything else moves to a server.

There is a workaround: export your browser cookies and give them to yt-dlp, which makes the
requests look like a signed-in user. Our downloader already supports it
(`YTDLP_COOKIES_FROM_BROWSER`). But be clear-eyed about what that means — **a live session for a
Google account would be sitting on a server.** Anyone with access to that box can act as that
account. If you do it, use a throwaway Google account, never a personal or school one.

My honest recommendation: **do not put cookies on a server.** Download at home, where it just
works, and host everything else.

---

## 3. Three ways to do this

### Option A — split it (recommended)

Downloads stay on a home machine. Everything else moves off.

```
  HOME (any of us)              CLOUD (always on)
  ┌────────────────┐            ┌──────────────────────┐
  │ yt-dlp         │──uploads──▶│ object storage       │
  │ analysis (GPU) │            │  video + frames      │
  └────────────────┘            ├──────────────────────┤
          │                     │ Postgres (results)   │
          └────writes results──▶├──────────────────────┤
                                │ API + web app        │◀── all three of us
                                └──────────────────────┘
```

- **Nothing depends on one person's PC being awake** for viewing data. The website and the
  database are always up.
- Downloading and analysis are batch jobs. They can run when someone's machine is on, and the
  results are permanent.
- Cost: **$0–5/month.**

This is a real change to the code — right now the service assumes video files sit on its own
disk. It is the correct destination, but it is not a one-afternoon job.

### Option B — one cheap server, downloads still at home (simplest real step)

Skip the object storage for now. Put **Postgres in the cloud** and keep the service on a home
machine.

- Cost: **$0.**
- What you gain immediately: the *data* stops living on one laptop. If Justin's PC dies, no
  events are lost. Everyone works against one set of jobs, corrections and stats.
- What you do not gain: the website is only up when the host machine is up.

**This is the right first step**, because it is thirty minutes of work and gets you most of the
value. §5 explains exactly how.

### Option C — put everything on a server, cookies and all

Possible, not recommended. You inherit the bot-detection fight, you put a Google session on a
box, and you pay for CPU to do analysis that Robert's GPU does faster for free. Doc 2 also notes
bulk downloading is against YouTube's terms — that is a tolerable tradeoff on our own machines,
and a worse one on rented infrastructure under someone's name.

---

## 4. Where each kind of data lives

### Video segments — the big one

**Size:** a clipped 2:30 match at 1080p is **300–600 MB**. A full unclipped event VOD is
**15–25 GB** — we downloaded one by accident and it was 9.2 GB for five hours of footage.

**Where:** `data/segments/` on whichever machine downloads. In Option A, uploaded to object
storage afterwards.

**Back it up? No.** Doc 2: *"Downloaded media is a cache, not a record. It should be deletable at
any time without losing anything."* Deleted after analysis with a 7-day grace. If you need it
again, download it again.

### Extracted frames — the medium one

**Size:** at 2 fps a 2:30 match is ~300 frames. At `jpeg_quality: 85` that is roughly **45 MB per
match**, so 100 matches ≈ **4.5 GB**.

**Where:** `data/collections/<collection-id>/`, then Roboflow once they are reviewed.

**Back it up? No** — same reasoning. Frames are re-derivable from video with ffmpeg, which is why
the dataset stores `(video_id, start_offset, frame_number)` instead of pixels. 100k frames is
~15 MB as references versus 20–50 GB as JPEGs.

### The database — the small one that matters

**Size:** currently 64 KB. A full season of events is maybe **tens of megabytes**.

**Where:** SQLite file today, managed Postgres once shared.

**Back it up? Yes. This is the only irreplaceable thing we have.** Everything else can be
regenerated; a lost correction is a human's work gone. It is small enough that a scheduled copy to
Google Drive or OneDrive is completely fine — that is the one place Drive is the right tool,
because it is one small file rather than a hundred thousand tiny ones.

### Labels and the training dataset

**Where:** Roboflow free tier. It stores the images, the labels, the review UI and the exports.

### Summary

| Data | Size | Lives where | Backed up |
|---|---|---|---|
| Video segments | 300–600 MB each | Downloader's disk → object storage | No, it's a cache |
| Extracted frames | ~45 MB per match | `data/collections/` → Roboflow | No, re-derivable |
| **Database** | tens of MB | **Managed Postgres** | **Yes — the only thing that matters** |
| Labels | — | Roboflow | Roboflow holds it |
| Frame references | ~15 MB per 100k | In git | Yes, via git |

**Peak realistic footprint if you keep a week of an event:** roughly **30–50 GB** of video and
frames. That fits Oracle's free 200 GB, or costs about **$0.75/month** on Cloudflare R2.

---

## 5. Postgres, explained properly

### What it is and why we need it

SQLite — what we use now — is **one file on one disk**. That is why it cannot be shared: there is
no way for three computers to open the same file safely over the internet.

Postgres is a database **server**. It runs somewhere, listens on a port, and any number of
programs connect to it over the network at once. Same SQL, same tables, same code — the only
thing that changes is a connection string.

Doc 0 already anticipated this: *"Postgres in production, SQLite acceptable locally."* So this is
not a design change. It is doing what the contract already says.

### It is free at our size

Our database is tens of megabytes. Every managed provider's free tier is far more than we need:

- **[Neon](https://neon.tech)** — free tier, 0.5 GB, made for Postgres, simplest signup.
- **[Supabase](https://supabase.com)** — free tier, 500 MB, gives you a table browser in the
  web UI which is genuinely handy for poking at data.

Either is fine. Neon if you just want a database; Supabase if you want to click around in it.

### Setting it up

**1.** Sign up, create a project, and copy the connection string. It looks like:

```
postgresql://user:password@ep-something.aws.neon.tech/dbname?sslmode=require
```

**That string contains a password.** It goes in `ingest/.env` on the host machine and nowhere
else — not in git, not in chat, not in the repo. `.env` is already gitignored.

**2.** Put it in the host's `ingest/.env`:

```bash
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
```

**3.** Install dependencies and verify the connection:

```bash
# in: REPO
.\run.ps1 setup
.\run.ps1 db-check
```

**That's it.** The tables create themselves on first run. The service automatically loads
`ingest/.env`, reads `DATABASE_URL`, and only applies the SQLite-specific argument when the URL
starts with `sqlite`. `db-check` never echoes the password.

**4.** Confirm it worked: <http://localhost:8080/api/health> should return
`{"status":"ok","schema_version":3,...}`. If the connection string is wrong you will get a
connection error on startup instead, which is the failure you want — loud and immediate.

### Do we need to move the old data?

**No.** The current SQLite database only holds test jobs, and its events came from fixtures rather
than real analysis. Starting clean on Postgres loses nothing real and is much simpler than
writing a migration.

Once real matches have been analysed, that changes — at that point the database is the product
and you back it up rather than recreating it.

### What everyone else needs

**Nothing.** Doc 0 line 443: *"Component 2 owns the schema and is the only thing that writes to
it."* Only the machine running the ingest service ever touches the database. Everyone else opens
a URL in a browser.

Do not hand out the connection string. Not because anyone would abuse it, but because a password
in three places is a password in three chat logs, and there is no reason for it to be anywhere
but the host.

---

## 6. What I'd actually do, in order

## Team operating plan — use this now

This is the concrete answer for the hardware the team has. It intentionally separates a shared
**service of record** from the machines that do expensive work.

| Responsibility | Where it runs | Why |
|---|---|---|
| Shared results, corrections, jobs | **One Supabase Postgres project** | It is the small, irreplaceable data set. The free tier includes a dedicated Postgres database with a 500 MB limit, which is far beyond this project's expected records. |
| Downloading YouTube and the ingest service | **Justin's home PC, while it is needed** | Residential internet avoids the datacenter-IP YouTube problem. Justin already has the disk and can run the service. |
| C++ analysis | **Justin's home PC first; Robert's 3060 when GPU inference exists** | The current OpenCV proof is CPU work. Robert's NVIDIA GPU is the right first place for RF-DETR/ONNX work. |
| Detector training | **Robert's RTX 3060 12 GB** | CUDA works on Windows and 12 GB is enough for the planned 640px detector run. |
| Large, permissioned labeling batches | **Robotics-room PCs, after school only** | Use them only after a teacher authorizes software installation, offline input transfer, and after-hours use. They are workers, not a server. |
| Reviewed detector dataset | **Roboflow** | It is the review tool and dataset source. Do not use Drive as the live dataset filesystem. |
| Database backup | **One scheduled `pg_dump` file in Drive** | Drive is good for one small backup file; it is not good for hundreds of thousands of training images. |

**What to set up first:** Justin creates or is invited to one Supabase project; Robert is also an
owner. Put its pooled Postgres connection string only in the machine that runs ingest:

```text
DATABASE_URL=postgresql://...sslmode=require
```

That makes the database central immediately. It does **not** make the website permanently online:
the present code streams video from the same disk as the ingest service, so the site is available
only while that home worker is running. Tailscale is the safe way for teammates to reach it without
opening router ports. A 24/7 public site is a later project requiring object storage (R2 or similar)
and a worker/queue; do not put YouTube cookies on a rented server.

If the Supabase free project pauses after inactivity, wake it before an event and rely on the Drive
backup as the recovery path. The official free tier currently includes a 500 MB database and pauses
after a week of inactivity; check the provider's current limits before committing team money.

1. **Set up Neon or Supabase now** and put the connection string in the host's `.env`. Thirty
   minutes, free, and the data immediately stops depending on one laptop. This is Option B.
2. **Keep downloading and analysing at home.** It works, it is free, and it dodges the
   datacenter-IP problem entirely.
3. **Use Tailscale** so the others can reach the host from anywhere without exposing anything to
   the internet.
4. **Revisit Option A later**, once there is real data worth keeping up 24/7. Moving video to
   Cloudflare R2 — S3-compatible, ~$0.015/GB/month, and crucially **zero egress fees**, which
   matters when the player streams video — is the change that lets the service live anywhere. It
   is a real refactor, not a config change, so it should wait until the pipeline actually
   produces something.

The honest summary: **you can stop depending on one PC for your *data* today, for free. Not
depending on one PC for the *website* is a bigger job, and it is not worth doing until the
analysis backend produces something worth serving.**

## 7. Secrets, rotation, and the rate limit

Everything secret lives in environment variables — in the host's Render env for the deployed
service, and in `ingest/.env` locally. **None of it is ever committed** (`.gitignore` keeps `.env`,
weights, video, and service-account JSON out). If a secret lands in git history or a request log,
treat it as burned and rotate it.

**The secrets**

| Variable | What it is | Who needs it |
|---|---|---|
| `DATABASE_URL` | Postgres connection string | the hosted service only |
| `TENGEN_WRITE_KEY` | lets a client publish/edit (`X-Tengen-Key` header) | whoever publishes from the desktop app |
| `TENGEN_READ_KEY` | lets a client read | everyone who views the site |
| `SUPABASE_SERVICE_KEY` | signs the private video bucket's links | the hosted service only |

The media-token secret is **derived** from the write/read key, not stored separately — so rotating
a key also invalidates outstanding video tokens. That is fine: tokens are short-lived (6 h) and the
page just fetches a fresh one.

**What the server already enforces** (`ingest/hosted.py`)

- With `TENGEN_REQUIRE_KEYS=1`, a server with no keys **closes** (503) instead of opening to anyone.
- Read vs write split: the read key cannot publish or edit.
- Keys travel only in the `X-Tengen-Key` header, never in URLs (URLs end up in logs).
- Video URLs carry a **signed, expiring, per-job token**, not the key.
- `TENGEN_PUBLISH_ONLY=1` makes the server serve published runs but refuse to run analysis.
- **Per-IP rate limiting.** On a hosted server (keys required) each IP gets `TENGEN_RATE_LIMIT`
  requests per minute — default **240**, generous for the UI but a cap on a scraper. Set
  `TENGEN_RATE_LIMIT=0` to disable, or any number to tune. It is off for a local install
  (health checks are always exempt, so monitors never trip it).

**Rotating a key**

1. Generate a new random value (e.g. `python -c "import secrets; print(secrets.token_urlsafe(24))"`).
2. Update it in the host's Render env and redeploy.
3. Update `ingest/.env` and `data/tengen-server.json` on each machine that publishes.
4. Re-share the new **read** key with viewers (the write key stays with publishers only).

Rotate immediately after any suspected exposure, and as a habit at the **start of each season**.
