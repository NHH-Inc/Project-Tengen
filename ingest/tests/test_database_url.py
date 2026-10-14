"""The server must start whichever common form of a Postgres address it is given."""

import importlib.util
import unittest

from ingest.database import normalise_url

PSYCOPG3 = importlib.util.find_spec("psycopg") is not None


class NormaliseUrlTests(unittest.TestCase):
    def test_heroku_style_scheme(self):
        self.assertEqual(normalise_url("postgres://u:p@h:5432/db"), "postgresql://u:p@h:5432/db")

    @unittest.skipIf(PSYCOPG3, "psycopg 3 is installed here, so its URL is left alone")
    def test_psycopg3_url_falls_back_to_the_installed_driver(self):
        self.assertEqual(normalise_url("postgresql+psycopg://u:p@h:6543/postgres"),
                         "postgresql+psycopg2://u:p@h:6543/postgres")

    def test_other_urls_are_untouched(self):
        for url in ("postgresql://u:p@h/db", "postgresql+psycopg2://u:p@h/db", "sqlite://"):
            self.assertEqual(normalise_url(url), url)


if __name__ == "__main__":
    unittest.main()
