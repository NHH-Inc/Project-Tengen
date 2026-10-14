import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# Make direct imports (for example ``python -m ingest.db_check``) honour the same host-only
# configuration as the FastAPI app.
from . import settings  # noqa: F401
from .models import Base

# Doc 0: "Postgres in production, SQLite acceptable locally." Default to the local SQLite
# file; point DATABASE_URL at Postgres (:5432 per doc 0's defaults) in production.
def normalise_url(url: str) -> str:
    """Accept every common way of writing a Postgres address, and use the driver installed here.

    Supabase's dashboard offers `postgresql+psycopg://` (psycopg 3) for SQLAlchemy, Heroku-style
    hosts say `postgres://`, and requirements.txt installs psycopg2. Pasting the dashboard's string
    into a hosting provider crashed the server at import with "No module named 'psycopg'".
    """
    import importlib.util

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql+psycopg://") and importlib.util.find_spec("psycopg") is None:
        url = "postgresql+psycopg2://" + url[len("postgresql+psycopg://"):]
    return url


SQLALCHEMY_DATABASE_URL = normalise_url(os.environ.get("DATABASE_URL", "sqlite:///./frc_scouting.db"))

# check_same_thread is a SQLite-only argument; passing it to Postgres raises.
connect_args = (
    {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=connect_args,
    # Managed Postgres can briefly drop idle connections when waking from its free tier.
    # Pre-ping replaces a stale pooled connection before a job gets far enough to fail.
    pool_pre_ping=not SQLALCHEMY_DATABASE_URL.startswith("sqlite"),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)
    # create_all does not alter pre-existing tables. Upgrade the team's existing jobs table
    # in place; this optional field is backward compatible with Contract A.
    columns = {column["name"] for column in inspect(engine).get_columns("jobs")}
    if "capture_mode" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE jobs ADD COLUMN capture_mode VARCHAR DEFAULT 'recorded'")
            )

    # Anything else that drifted gets named out loud. Without this, a model change that nobody
    # backfills surfaces as `column jobs.<x> does not exist` in the middle of an unrelated
    # query, which reads like a code bug rather than a schema one. Adding a column above is a
    # one-line fix once you know which column; finding out is the expensive part.
    for table, missing in schema_drift().items():
        print(
            f"WARNING: table '{table}' is missing column(s) {', '.join(missing)} that the models "
            "declare. Queries touching them will fail. Add an ALTER TABLE in init_db(), or drop "
            "the table if it holds nothing you need. See docs/RUNNING.md.",
            flush=True,
        )


def schema_drift() -> dict[str, list[str]]:
    """Columns the models declare that the live database does not have.

    ``create_all()`` only creates missing TABLES -- it never adds a column to a table that
    already exists. Every model change therefore desynchronises every database that predates
    it, and the shared Postgres is the one that hurts.
    """
    inspector = inspect(engine)
    live_tables = set(inspector.get_table_names())
    drift: dict[str, list[str]] = {}
    for table in Base.metadata.sorted_tables:
        if table.name not in live_tables:
            continue  # create_all handles a wholly missing table correctly
        live = {column["name"] for column in inspector.get_columns(table.name)}
        if missing := sorted({c.name for c in table.columns} - live):
            drift[table.name] = missing
    return drift


def verify_connection() -> str:
    """Make a cheap real query and return only the database dialect, never its URL."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return engine.url.get_backend_name()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
