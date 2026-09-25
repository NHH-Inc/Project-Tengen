"""Segment retention.

Doc 2: "Downloaded media is a cache, not a record. It should be deletable at any time without
losing anything: the event data extracted from it is the actual product and lives in the
database. Set a retention policy early or disk usage will get out of hand fast."

DECISIONS D9 settled the policy -- delete a segment once its job is complete, keeping a grace
window for re-analysis -- but nothing implemented it until now, so segments accumulated forever.

Nothing here touches events, tracks or corrections. Only cached media.
"""

import datetime
import os
from pathlib import Path

from . import models

#: Keep a completed job's segment this long before reclaiming it, so a re-run does not have to
#: re-download. Zero deletes as soon as analysis finishes.
GRACE_DAYS = float(os.environ.get("FRC_SEGMENT_GRACE_DAYS", "7"))


def sweep(db, grace_days: float | None = None, dry_run: bool = False) -> dict:
    """Delete cached segments for completed jobs older than the grace window.

    Returns a summary rather than logging, so the caller decides how loud to be.
    """
    grace = GRACE_DAYS if grace_days is None else grace_days
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=grace)

    freed_bytes = 0
    removed = []
    for job in db.query(models.Job).filter(models.Job.status == "complete").all():
        if not job.local_path:
            continue
        # An uploaded recording is not a cache. The premise above -- media can be fetched again
        # -- holds for a YouTube segment and fails for a phone recording, where this file is the
        # only copy the app has.
        if job.capture_mode == "local":
            continue
        updated = job.updated_at
        if updated is not None and updated.tzinfo is None:
            updated = updated.replace(tzinfo=datetime.timezone.utc)
        if updated is not None and updated > cutoff:
            continue  # still inside the grace window

        path = Path(job.local_path)
        if not path.exists():
            job.local_path = None  # already gone; stop pointing at nothing
            continue

        size = path.stat().st_size
        if dry_run:
            freed_bytes += size
            removed.append(job.job_id)
            continue
        try:
            path.unlink()
        except OSError:
            continue
        # The job stays. Its events are the product; only the cache is gone.
        job.local_path = None
        freed_bytes += size
        removed.append(job.job_id)

    if not dry_run:
        db.commit()
    return {
        "jobs": len(removed),
        "freed_gb": round(freed_bytes / (1024 ** 3), 2),
        "grace_days": grace,
        "dry_run": dry_run,
    }


def _main() -> int:
    """CLI so this is runnable without touching Contract E.

        python -m ingest.retention --dry-run
        python -m ingest.retention

    Deliberately not an API endpoint: Contract E is the shared surface component 3 builds
    against, and adding to it needs all three people. Housekeeping does not belong there.
    """
    import argparse

    from . import database

    parser = argparse.ArgumentParser(description="Delete cached segments for completed jobs.")
    parser.add_argument("--grace-days", type=float, default=None,
                        help=f"override the grace window (default {GRACE_DAYS})")
    parser.add_argument("--dry-run", action="store_true", help="report without deleting")
    args = parser.parse_args()

    db = next(database.get_db())
    try:
        result = sweep(db, grace_days=args.grace_days, dry_run=args.dry_run)
    finally:
        db.close()

    verb = "would free" if result["dry_run"] else "freed"
    print(f"{result['jobs']} segment(s), {verb} {result['freed_gb']} GB "
          f"(grace {result['grace_days']} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
