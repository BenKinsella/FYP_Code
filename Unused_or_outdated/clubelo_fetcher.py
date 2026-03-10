import os
import sys
import psycopg2
import requests
import logging
import time
import csv
from io import StringIO
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class EloTableUpdater:
    def __init__(self, database_url):
        self.database_url = database_url
        self.conn = None
        self.elo_cache = {}

        # Set up session with retry logic (1)
        self.session = requests.Session()
        retry_strategy = Retry(
            total=4,                  # Up to 4 retries on connection errors
            backoff_factor=2,         # Delay pattern: 2s, 4s, 8s, 16s...
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)


    def connect_db(self):
        try:
            self.conn = psycopg2.connect(self.database_url)
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            sys.exit(1)


    def close_db(self):
        if self.conn:
            self.conn.close()


    def fetch_pinnacle_events(self):
        try:
            with self.conn.cursor() as cursor:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=3)
                cursor.execute("""
                    SELECT event_id, logged_time, home_team, away_team, league_id, league_name, home_odds, draw_odds, away_odds, starts
                    FROM odds1x2
                    WHERE starts < %s
                """, (cutoff_time,))
                rows = cursor.fetchall()
            return [
                {
                    "event_id": r[0],
                    "logged_time": r[1],
                    "home_team": r[2].strip().lower() if r[2] else "",
                    "away_team": r[3].strip().lower() if r[3] else "",
                    "league_id": r[4],
                    "league_name": r[5],
                    "home_odds": r[6],
                    "draw_odds": r[7],
                    "away_odds": r[8],
                    "starts": r[9],
                } for r in rows
            ]
        except Exception as e:
            logger.error(f"Error fetching Pinnacle events: {e}")
            return []


    def fetch_clubelo_data_for_date(self, date_str):
        """Fetch ClubELO data for one day with retry and delay."""
        if date_str in self.elo_cache:
            return self.elo_cache[date_str]

        url = f"https://api.clubelo.com/{date_str}"

        # Throttle requests slightly (2)
        time.sleep(1.5)

        try:
            logger.info(f"Fetching ClubELO data for {date_str}")
            response = self.session.get(url, timeout=120)
            response.raise_for_status()
            reader = csv.DictReader(StringIO(response.text))
            elo_map = {}
            for row in reader:
                team = row.get("Club", "").strip().lower()
                elo = row.get("Elo")
                if team and elo:
                    elo_map[team] = float(elo)
            self.elo_cache[date_str] = elo_map
            return elo_map

        except Exception as e:
            logger.warning(f"Failed to fetch ClubELO for {date_str}: {e}")
            self.elo_cache[date_str] = {}
            return {}


    def _prefix3(self, name: str) -> str:
        if not isinstance(name, str):
            return ""
        return name.strip().lower()[:3]


    def _words(self, name: str) -> set:
        if not isinstance(name, str):
            return set()
        cleaned = name.replace(".", " ").replace("'", " ")
        return set(w for w in cleaned.lower().split() if w)


    def match_and_insert_results(self, matches):
        insert_query = """
            INSERT INTO elo_table (
                event_id, logged_time, home_team, away_team,
                league_id, league_name, home_odds, draw_odds, away_odds, starts, home_ELO, away_ELO, ELO_diff
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """
        with self.conn.cursor() as cursor:
            count_inserted = 0
            for m in matches:
                date_str = m["logged_time"].strftime("%Y-%m-%d")
                elo_map = self.fetch_clubelo_data_for_date(date_str)

                home_elo = elo_map.get(m["home_team"])
                away_elo = elo_map.get(m["away_team"])

                if home_elo is None or away_elo is None:
                    logger.debug(f"Missing ELO for one or both teams: {m['home_team']} vs {m['away_team']} on {date_str}")
                    continue

                elo_diff = home_elo - away_elo

                try:
                    cursor.execute(insert_query, (
                        m["event_id"], m["logged_time"], m["home_team"], m["away_team"],
                        m["league_id"], m["league_name"], m["home_odds"], m["draw_odds"],
                        m["away_odds"], m["starts"], home_elo, away_elo, elo_diff
                    ))
                    count_inserted += 1
                except Exception as e:
                    logger.warning(f"Insert failed for {m['event_id']} on {date_str}: {e}")

            self.conn.commit()
            logger.info(f"Inserted {count_inserted} elo_table records.")


    def update(self):
        self.connect_db()
        try:
            matches = self.fetch_pinnacle_events()
            unique_days = len(set([m['logged_time'].date() for m in matches]))
            logger.info(f"Processing {len(matches)} odds logs across {unique_days} days.")
            self.match_and_insert_results(matches)
        finally:
            self.close_db()


if __name__ == "__main__":
    DATABASE_URL = os.environ["DATABASE_URL"]
    updater = EloTableUpdater(DATABASE_URL)
    updater.update()
