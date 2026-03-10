import os
import psycopg2
import requests
from datetime import datetime

class ResultsUpdater:
    def __init__(self, database_url, league_id, api_key='1'):
        self.database_url = database_url
        self.league_id = league_id
        self.api_key = api_key
        self.conn = None

    def connect_db(self):
        self.conn = psycopg2.connect(self.database_url)

    def close_db(self):
        if self.conn:
            self.conn.close()

    def fetch_past_pinnacle_events(self):
        with self.conn.cursor() as cursor:
            cursor.execute("""
              SELECT event_id, home_team, away_team, starts
              FROM odds1x2
              WHERE starts < NOW()
            """)
            return cursor.fetchall()

    def fetch_finished_events(self):
        url = f"https://www.thesportsdb.com/api/v1/json/{self.api_key}/eventspastleague.php?id={self.league_id}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("event") or data.get("events") or []

    def match_and_insert_results(self, pinnacle_events, thesportsdb_events):
        insert_query = """
            INSERT INTO results (home_team, away_team, starts, home_score, away_score)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """
        count = 0
        with self.conn.cursor() as cursor:
            for p_event in pinnacle_events:
                p_event_id, p_home, p_away, p_starts = p_event
                p_starts_dt = p_starts if isinstance(p_starts, datetime) else datetime.fromisoformat(str(p_starts))
                for db_event in thesportsdb_events:
                    if (
                        db_event.get("strStatus") == "Match Finished"
                        and db_event.get("strHomeTeam") and db_event.get("strAwayTeam")
                        and db_event.get("intHomeScore") is not None
                        and db_event.get("intAwayScore") is not None
                    ):
                        db_home = db_event.get("strHomeTeam").strip().lower()
                        db_away = db_event.get("strAwayTeam").strip().lower()
                        p_home_norm = p_home.strip().lower()
                        p_away_norm = p_away.strip().lower()

                        # Parse event start time from TheSportsDB
                        db_ts = db_event.get("strTimestamp")
                        if db_ts:
                            try:
                                db_starts_dt = datetime.fromisoformat(db_ts)
                            except Exception:
                                db_starts_dt = None
                        else:
                            db_date = db_event.get("dateEvent")
                            db_time = db_event.get("strTime") or "00:00:00"
                            try:
                                db_starts_dt = datetime.fromisoformat(f"{db_date}T{db_time}")
                            except Exception:
                                db_starts_dt = None

                        # Allow up to 2 hours difference for time zone safety
                        if (
                            db_home == p_home_norm
                            and db_away == p_away_norm
                            and db_starts_dt is not None
                            and abs((db_starts_dt - p_starts_dt).total_seconds()) < 2 * 3600
                        ):
                            cursor.execute(insert_query, (
                                p_home,
                                p_away,
                                p_starts_dt,
                                int(db_event.get("intHomeScore")),
                                int(db_event.get("intAwayScore"))
                            ))
                            count += 1
                            break
            self.conn.commit()
        print(f"Inserted {count} matched finished matches for this league.")

    def update(self):
        self.connect_db()
        try:
            pinnacle_events = self.fetch_past_pinnacle_events()
            thesportsdb_events = self.fetch_finished_events()
            self.match_and_insert_results(pinnacle_events, thesportsdb_events)
        finally:
            self.close_db()

if __name__ == "__main__":
    DATABASE_URL = os.environ["DATABASE_URL"]
    SPORTAPI_KEY = os.environ.get("SPORTAPI_KEY", "1")
    league_ids = [
        ("English Premier League", "4328"),
        ("Spanish La Liga", "4335"),
        ("German Bundesliga", "4331"),
        ("Italian Serie A", "4332"),
        ("French Ligue 1", "4334"),
    ]
    for league_name, league_id in league_ids:
        print(f"\nUpdating results for: {league_name} (League ID: {league_id})")
        updater = ResultsUpdater(DATABASE_URL, league_id, SPORTAPI_KEY)
        updater.update()
