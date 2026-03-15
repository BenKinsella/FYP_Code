import os
import psycopg2
import requests
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

# Set up logging as in your odds scraper
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ResultsUpdaterSportAPI:
    def __init__(self, database_url, tournament_id, season_id, api_key):
        self.database_url = database_url
        self.tournament_id = tournament_id
        self.season_id = season_id
        self.api_key = api_key
        self.conn = None

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
                LEAGUE_ID = 2406
                cursor.execute("""
                    SELECT event_id, home_team, away_team, league_id, league_name, starts
                    FROM odds1x2
                    WHERE starts < %s
                    AND league_id = %s
                """, (cutoff_time, LEAGUE_ID))
                rows = cursor.fetchall()
            return [
                {
                    "event_id": r[0],
                    "home_team": r[1].strip().lower() if r[1] else "",
                    "away_team": r[2].strip().lower() if r[2] else "",
                    "league_id": r[3],
                    "league_name": r[4],
                    "starts": r[5]
                } for r in rows
            ]
        except Exception as e:
            logger.error(f"Error fetching Pinnacle events: {e}")
            return []

    def fetch_sportapi_events_page(self, tournament_id, season_id, page=1):
        url = f"https://footapi7.p.rapidapi.com/api/tournament/203/season/77142/matches/last/1"
        headers = {
            "X-RapidAPI-Key": self.api_key

            # "X-RapidAPI-Host": self.api_host
        }
        try:
            logger.info(f"Requesting events from FootAPI")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json().get("events", [])
        except Exception as e:
            logger.error(f"API error fetching events: {e}")
            return []

    def _prefix3(self, name: str) -> str:
        if not isinstance(name, str):
            return ""
        return name.strip().lower()[:3]

    def _words(self, name: str) -> set:
        if not isinstance(name, str):
            return set()
        cleaned = name.replace(".", " ").replace("'", " ")
        return set(w for w in cleaned.lower().split() if w)

    def match_and_insert_results(self, pinnacle_events, sportapi_events):
        insert_query = """
            INSERT INTO results1 (
                event_id, home_team, away_team,
                league_id, league_name, starts,
                home_score, away_score, result
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """
        count = 0
        skipped = 0

        with self.conn.cursor() as cursor:
            for api_event in sportapi_events:
                status_type = api_event.get("status", {}).get("type")
                if status_type != "finished":
                    skipped += 1
                    continue

                if str(api_event.get("tournament", {}).get("id")) not in [
                    str(tid) for tid in self.tournament_id
                ]:
                    skipped += 1
                    continue

                home_score = api_event.get("homeScore", {}).get("normaltime")
                away_score = api_event.get("awayScore", {}).get("normaltime")
                if home_score is None or away_score is None:
                    skipped += 1
                    continue

                home_data = api_event.get("homeTeam", {}) or {}
                away_data = api_event.get("awayTeam", {}) or {}

                api_home_short = (home_data.get("shortName") or "").strip().lower()
                api_home_name  = (home_data.get("name") or "").strip().lower()
                api_home_code  = (home_data.get("nameCode") or "").strip().lower()

                api_away_short = (away_data.get("shortName") or "").strip().lower()
                api_away_name  = (away_data.get("name") or "").strip().lower()
                api_away_code  = (away_data.get("nameCode") or "").strip().lower()

                api_start = datetime.fromtimestamp(
                    api_event.get("startTimestamp"),
                    tz=timezone.utc,
                )

                matched = False

                for p_event in pinnacle_events:
                    p_home = p_event["home_team"]
                    p_away = p_event["away_team"]

                    logger.info(
                        f"Comparing Pinnacle: '{p_home}' vs '{p_away}' @ {p_event['starts']} "
                        f"with API: home='{api_home_name}', away='{api_away_name}' @ {api_start}"
                    )

                    # ---------- strict match: exact names/codes + date window ----------
                    home_matches = (
                        api_home_short == p_home
                        or api_home_short == p_away
                        or api_home_name == p_home
                        or api_home_name == p_away
                        or api_home_code == p_home
                        or api_home_code == p_away
                    )
                    away_matches = (
                        api_away_short == p_away
                        or api_away_short == p_home
                        or api_away_name == p_away
                        or api_away_name == p_home
                        or api_away_code == p_away
                        or api_away_code == p_home
                    )

                    if home_matches and away_matches and \
                    abs((api_start.date() - p_event["starts"].date()).days) <= 2:

                        result = (
                            "home_win" if home_score > away_score
                            else "away_win" if away_score > home_score
                            else "draw"
                        )

                        try:
                            cursor.execute(
                                insert_query,
                                (
                                    p_event["event_id"],
                                    p_event["home_team"],
                                    p_event["away_team"],
                                    p_event["league_id"],
                                    p_event["league_name"],
                                    api_start,
                                    home_score,
                                    away_score,
                                    result,
                                ),
                            )
                            count += 1
                            logger.info(f"Inserted result for event_id {p_event['event_id']}")
                        except Exception as e:
                            logger.error(f"DB insert error: {e}")
                        matched = True
                        break

                    # ---------- fallback: prefix or word match + same calendar day ----------
                    same_day = api_start.date() == p_event["starts"].date()
                    if not same_day:
                        continue

                    api_home_prefix = self._prefix3(api_home_name or api_home_short or api_home_code)
                    api_away_prefix = self._prefix3(api_away_name or api_away_short or api_away_code)
                    p_home_prefix   = self._prefix3(p_home)
                    p_away_prefix   = self._prefix3(p_away)

                    prefix_ok = (
                        (api_home_prefix and api_home_prefix == p_home_prefix)
                        or (api_home_prefix and api_home_prefix == p_away_prefix)
                        or (api_away_prefix and api_away_prefix == p_home_prefix)
                        or (api_away_prefix and api_away_prefix == p_away_prefix)
                    )

                    api_home_words = self._words(api_home_name or api_home_short or api_home_code)
                    api_away_words = self._words(api_away_name or api_away_short or api_away_code)
                    p_home_words   = self._words(p_home)
                    p_away_words   = self._words(p_away)

                    home_word_ok = len(api_home_words & p_home_words) > 0 or \
                                len(api_home_words & p_away_words) > 0
                    away_word_ok = len(api_away_words & p_away_words) > 0 or \
                                len(api_away_words & p_home_words) > 0

                    word_ok = home_word_ok or away_word_ok

                    if (prefix_ok or word_ok) and same_day:
                        result = (
                            "home_win" if home_score > away_score
                            else "away_win" if away_score > home_score
                            else "draw"
                        )

                        try:
                            cursor.execute(
                                insert_query,
                                (
                                    p_event["event_id"],
                                    p_event["home_team"],
                                    p_event["away_team"],
                                    p_event["league_id"],
                                    p_event["league_name"],
                                    api_start,
                                    home_score,
                                    away_score,
                                    result,
                                ),
                            )
                            count += 1
                            logger.info(
                                f"Inserted result via fallback for event_id {p_event['event_id']}"
                            )
                        except Exception as e:
                            logger.error(f"DB insert error (fallback): {e}")
                        matched = True
                        break

                if not matched:
                    logger.debug(
                        f"No Pinnacle match found for API event: "
                        f"home='{api_home_name}', away='{api_away_name}' @ {api_start}"
                    )
                    skipped += 1

        try:
            self.conn.commit()
        except Exception as e:
            logger.error(f"DB commit error: {e}")
        logger.info(f"Inserted {count} matches, skipped {skipped} events in this batch.")

        logger.info(f"Example Pinnacle row: {pinnacle_events[0]}")
        logger.info(f"Example FootAPI event: {sportapi_events[0]['homeTeam']['name']} vs {sportapi_events[0]['awayTeam']['name']} @ {datetime.fromtimestamp(sportapi_events[0]['startTimestamp'], tz=timezone.utc)}")

    def update(self):
        self.connect_db()
        try:
            pinnacle_events = self.fetch_pinnacle_events()

            # Single call – no paging
            all_events = self.fetch_sportapi_events_page(
                tournament_id=self.tournament_id[0],
                season_id=self.season_id,
                page=1,
            )

            pinnacle_events = self.fetch_pinnacle_events()
            logger.info(f"Fetched {len(pinnacle_events)} Pinnacle events")


            logger.info(f"Processing SportAPI events: found {len(all_events)} events.")
            self.match_and_insert_results(pinnacle_events, all_events)
        finally:
            self.close_db()

if __name__ == "__main__":
    DATABASE_URL = os.environ.get("DATABASE_URL")
    SPORTAPI_KEY = os.environ.get("SPORTAPI_KEY")
    # SPORTAPI_HOST = os.environ.get("SPORTAPI_HOST", "sportapi7.p.rapidapi.com")
    TOURNAMENT_IDS = [203]  # Premier League ID for SportAPI
    SEASON_ID = 77142  

    # start_date = datetime.strptime("2025-11-10", "%Y-%m-%d").date()
    # end_date = datetime.strptime("2025-11-10", "%Y-%m-%d").date()

    updater = ResultsUpdaterSportAPI(
        database_url=DATABASE_URL,
        api_key=SPORTAPI_KEY,
        # api_host=SPORTAPI_HOST,
        tournament_id=TOURNAMENT_IDS,
        season_id=SEASON_ID,
        # start_date=start_date,
        # end_date=end_date,
    )
    updater.update()