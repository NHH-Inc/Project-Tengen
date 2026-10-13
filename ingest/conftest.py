"""Keep the test suite away from the real database, the real data folder, and the smoke script.

`ingest.settings` loads `ingest/.env` without overriding what is already set, and
`ingest.database` connects -- and runs `init_db()`, which can ALTER a table -- the moment
`ingest.main` is imported. Several test modules set a throwaway DATABASE_URL before that import,
but not all of them, so whichever module pytest happened to import first decided whether the
suite talked to SQLite or to the team's shared Postgres. pytest imports this file before any test
module below it, so these defaults always win that race. They are defaults: a run that sets its
own values explicitly still gets them.

`smoke_test.py` is a script (`python -m ingest.smoke_test`), but its name matches pytest's
`*_test.py` pattern, so `pytest ingest` imported it. At import it repoints the database and, to
stay offline, replaces `main.process_job` with a no-op -- for every test in the same process. The
upload tests then saw their job never run, and failed only when the run was pointed at `ingest/`
rather than `ingest/tests/`.
"""

import os
import tempfile

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("FRC_DATA_DIR", tempfile.mkdtemp(prefix="frc-scouting-tests-"))

collect_ignore = ["smoke_test.py"]
