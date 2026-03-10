import psycopg2
import os
import pandas as pd
import soccerdata as sd
import logging
from datetime import datetime, timedelta, timezone


class ResultsUpdaterSoccerdata:
    def __init__(self, database_url, league, season):
        self.database_url = database_url
        self.league = league
        self.season = season
        self.conn = None

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ResultsUpdaterSoccerdata")

    # ---------- DB helpers ----------
    def connect_db(self):
        if self.conn is None:
            self.conn = psycopg2.connect(self.database_url)

    def close_db(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # ---------- Data fetchers ----------
    def fetch_completed_odds_events(self):
        """Fetch finished matches from odds1x2; no scores here."""
        self.connect_db()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=3)
        LEAGUE_ID = 1980
        query = """
            SELECT event_id, home_team, away_team, league_id, league_name, starts
            FROM odds1x2
            WHERE starts < %s
            AND league_id = %s
        """
        df = pd.read_sql_query(query, self.conn, params=(cutoff,LEAGUE_ID))
        # Normalised names for matching
        for col in ["home_team", "away_team", "league_name"]:
            df[col + "_lc"] = df[col].astype(str).str.strip().str.lower()
        df["starts_date"] = pd.to_datetime(df["starts"]).dt.date
        # Deduplicate by event_id to avoid multiple inserts
        df = df.sort_values("starts").drop_duplicates(subset=["event_id"])
        return df

    '''
    def fetch_fbref_results(self):
        """Download schedule/results from FBref via soccerdata."""
        fbref = sd.FBref(self.league, self.season)
        fbref_results = fbref.read_schedule()

        fbref_results["date"] = pd.to_datetime(fbref_results["date"])
        fbref_results["home_team_lc"] = fbref_results["home_team"].str.strip().str.lower()
        fbref_results["away_team_lc"] = fbref_results["away_team"].str.strip().str.lower()
        fbref_results["date_only"] = fbref_results["date"].dt.date

        # Parse 'score' column like '2–1' into integers
        def parse_score(s):
            if not isinstance(s, str) or "–" not in s:
                return None, None
            parts = s.split("–")
            try:
                return int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                return None, None

        fbref_results[["home_score_fb", "away_score_fb"]] = fbref_results["score"].apply(
            lambda s: pd.Series(parse_score(s))
        )

        return fbref_results
    '''

    '''
    def fetch_fotmob_results(self):
        """Download schedule/results from FotMob via soccerdata."""
        fotmob = sd.FotMob(self.league, self.season)
        fotmob_results = fotmob.read_matches()   # check docs / print head()

        # Standardise columns to match fbref-based logic
        # Adjust these column names to FotMob's schema
        fotmob_results["date"] = pd.to_datetime(fotmob_results["date"])
        fotmob_results["home_team_lc"] = fotmob_results["home_team"].str.strip().str.lower()
        fotmob_results["away_team_lc"] = fotmob_results["away_team"].str.strip().str.lower()
        fotmob_results["date_only"] = fotmob_results["date"].dt.date

        # If FotMob has separate goal columns e.g. 'home_score', 'away_score'
        fotmob_results["home_score_fb"] = fotmob_results["home_score"]
        fotmob_results["away_score_fb"] = fotmob_results["away_score"]

        return fotmob_results
    '''

    def fetch_espn_results(self):
        """Download schedule/results from ESPN via soccerdata."""
        espn = sd.ESPN(leagues=self.league, seasons=self.season)
        schedule = espn.read_schedule()  # fixtures + meta

        print(schedule.columns)
        print(schedule.head())


        # Inspect once to confirm columns:
        # print(schedule.columns); print(schedule.head())

        # Standardise to your existing schema
        schedule["date"] = pd.to_datetime(schedule["date"])
        schedule["home_team"] = schedule["home_team"]
        schedule["away_team"] = schedule["away_team"]

        schedule["home_team_lc"] = schedule["home_team"].astype(str).str.strip().str.lower()
        schedule["away_team_lc"] = schedule["away_team"].astype(str).str.strip().str.lower()
        schedule["date_only"] = schedule["date"].dt.date

        # ESPN schedule usually does NOT contain final scores, so we may need matchsheet.
        # Let's assume schedule has 'game_id' and use it to pull scores.
        '''
        def get_scores(row):
            matchsheet = espn.read_matchsheet(match_id=row["game_id"])
            # matchsheet is team-level, with is_home flag and 'total_goals' per team. [web:26]
            home_row = matchsheet[matchsheet["is_home"] == True]
            away_row = matchsheet[matchsheet["is_home"] == False]
            if home_row.empty or away_row.empty:
                return pd.Series({"home_score_fb": None, "away_score_fb": None})
            return pd.Series(
                {
                    "home_score_fb": int(home_row["total_goals"].iloc[0]),
                    "away_score_fb": int(away_row["total_goals"].iloc[0]),
                }
            )
        '''
        def get_scores(row):
            try:
                # Pass match_id as list (fixes TypeError)
                ms = espn.read_matchsheet(match_id=[row["game_id"]])
                
                # Print ONCE to debug structure, then comment out
                # print(f"Game {row['game_id']}: columns={ms.columns.tolist()}")
                # print(f"Game {row['game_id']}: index names={ms.index.names}")
                # print(ms.head())
                
                # Skip .xs() entirely - work with full matchsheet
                # Try common goal columns (pick the right one after debugging)
                goal_cols = ['goals', 'total_goals', 'g', 'score']
                team_cols = ['team', 'team_name', 'club']
                
                goal_col = next((col for col in goal_cols if col in ms.columns), None)
                team_col = next((col for col in team_cols if col in ms.columns), None)
                
                if goal_col is None or team_col is None:
                    return pd.Series({"home_score_fb": None, "away_score_fb": None})
                
                # Group by team column and sum goals
                team_goals = ms.groupby(team_col)[goal_col].sum().reset_index()
                
                home_name = row["home_team"]
                away_name = row["away_team"]
                
                home_goals = team_goals[team_goals[team_col] == home_name]
                away_goals = team_goals[team_goals[team_col] == away_name]
                
                if home_goals.empty or away_goals.empty:
                    return pd.Series({"home_score_fb": None, "away_score_fb": None})
                    
                return pd.Series({
                    "home_score_fb": int(home_goals[goal_col].iloc[0]),
                    "away_score_fb": int(away_goals[goal_col].iloc[0])
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to get scores for game {row['game_id']}: {e}")
                return pd.Series({"home_score_fb": None, "away_score_fb": None})


        scores = schedule.apply(get_scores, axis=1)
        schedule = pd.concat([schedule, scores], axis=1)

        return schedule


    def fetch_matchhistory_results(self):
        """Download schedule/results from MatchHistory via soccerdata."""
        mh = sd.MatchHistory(leagues=self.league, seasons=self.season)
        hist = mh.read_games()  # historic match results + odds

        # Inspect once to confirm column names:
        print(hist.columns); print(hist.head())

        # Typical MatchHistory columns include 'HomeTeam', 'AwayTeam', 'Date', 'FTHG', 'FTAG' etc.
        # Adapt these if yours differ.
        hist["date"] = pd.to_datetime(hist["Date"])
        hist["home_team"] = hist["HomeTeam"]
        hist["away_team"] = hist["AwayTeam"]

        hist["home_team_lc"] = hist["home_team"].astype(str).str.strip().str.lower()
        hist["away_team_lc"] = hist["away_team"].astype(str).str.strip().str.lower()
        hist["date_only"] = hist["date"].dt.date

        # Full‑time goals as scores
        hist["home_score_fb"] = hist["FTHG"].astype(int)
        hist["away_score_fb"] = hist["FTAG"].astype(int)

        return hist



    # ---------- Business logic ----------
    def get_result(self, home_score, away_score):
        if home_score > away_score:
            return "home_win"
        elif home_score < away_score:
            return "away_win"
        else:
            return "draw"

    def update_results(self):
        odds_df = self.fetch_completed_odds_events()
        ## results_df = self.fetch_fbref_results()
        ## results_df = self.fetch_fotmob_results()
        ## results_df = self.fetch_matchhistory_results()
        results_df = self.fetch_espn_results()



        self.logger.info(f"Fetched {len(odds_df)} completed Pinnacle matches.")
        self.logger.info(f"Fetched {len(results_df)} matches from FBref.")

        # Prefix columns (first 3 letters) for fallback matching
        odds_df["home_prefix"] = odds_df["home_team_lc"].str[:3]
        odds_df["away_prefix"] = odds_df["away_team_lc"].str[:3]
        results_df["home_prefix"] = results_df["home_team_lc"].str[:3]
        results_df["away_prefix"] = results_df["away_team_lc"].str[:3]

        # ---------- 1) Strict merge: full team names + date ----------
        merged_strict = pd.merge(
            odds_df,
            results_df,
            left_on=["home_team_lc", "away_team_lc", "starts_date"],
            right_on=["home_team_lc", "away_team_lc", "date_only"],
            how="inner",
        )
        self.logger.info(f"Strictly matched {len(merged_strict)} matches (full names + date).")

        matched_event_ids = set(merged_strict["event_id"].unique())
        odds_unmatched = odds_df[~odds_df["event_id"].isin(matched_event_ids)].copy()

        # ---------- 2) Fallback 1: BOTH prefixes + date ----------
        fallback_prefix = pd.merge(
            odds_unmatched,
            results_df,
            left_on=["home_prefix", "away_prefix", "starts_date"],
            right_on=["home_prefix", "away_prefix", "date_only"],
            how="inner",
        )
        fallback_prefix = fallback_prefix.drop_duplicates(subset=["event_id"])
        self.logger.info(f"Fallback (home+away prefix) matched {len(fallback_prefix)} matches.")

        merged = pd.concat([merged_strict, fallback_prefix], ignore_index=True)
        merged = merged.drop_duplicates(subset=["event_id"])
        self.logger.info(f"Total matched {len(merged)} unique matches after prefix fallback.")

        # ---------- 3) Fallback 2: same date AND (word OR 3-letter) match for BOTH teams ----------
        matched_event_ids_all = set(merged["event_id"].unique())
        odds_unmatched2 = odds_df[~odds_df["event_id"].isin(matched_event_ids_all)].copy()

        if not odds_unmatched2.empty:
            # Candidate pairs: same date only
            candidates = odds_unmatched2.merge(
                results_df,
                left_on="starts_date",
                right_on="date_only",
                how="inner",
                suffixes=("_odds", "_fbref"),
            )

            def words(name: str) -> set:
                if not isinstance(name, str):
                    return set()
                return set(name.replace(".", " ").replace("'", " ").split())

            def name_soft_match(odds_name: str, fbref_name: str) -> bool:
                odds_name = odds_name or ""
                fbref_name = fbref_name or ""
                odds_lc = odds_name.strip().lower()
                fbref_lc = fbref_name.strip().lower()
                # 3-letter prefix match
                if odds_lc[:3] and odds_lc[:3] == fbref_lc[:3]:
                    return True
                # At least one common word
                return len(words(odds_lc) & words(fbref_lc)) > 0

            # Apply soft match for BOTH home and away
            mask_soft = candidates.apply(
                lambda r: name_soft_match(r["home_team_lc_odds"], r["home_team_lc_fbref"])
                or name_soft_match(r["away_team_lc_odds"], r["away_team_lc_fbref"]),
                axis=1,
            )

            fallback_soft = candidates[mask_soft].copy()
            fallback_soft = fallback_soft.drop_duplicates(subset=["event_id"])
            self.logger.info(f"Soft fallback (word OR 3-letter) matched {len(fallback_soft)} matches.")

            # For consistency with earlier merged frames,
            # rename odds-side columns back to *_x and fbref-side to *_y
            fallback_soft = fallback_soft.rename(
                columns={
                    "home_team_odds": "home_team_x",
                    "away_team_odds": "away_team_x",
                    "home_team_fbref": "home_team_y",
                    "away_team_fbref": "away_team_y",
                }
            )

            merged = pd.concat([merged, fallback_soft], ignore_index=True)
            merged = merged.drop_duplicates(subset=["event_id"])
            self.logger.info(f"Total matched {len(merged)} unique matches after soft fallback.")

        if len(merged) == 0:
            self.logger.warning("No matches to insert.")
            return

        # Keep only rows with valid parsed scores from FBref
        merged = merged.dropna(subset=["home_score_fb", "away_score_fb"])
        merged["home_score_fb"] = merged["home_score_fb"].astype(int)
        merged["away_score_fb"] = merged["away_score_fb"].astype(int)

        merged["result"] = merged.apply(
            lambda r: self.get_result(r["home_score_fb"], r["away_score_fb"]), axis=1
        )

        insert_query = """
            INSERT INTO results1 (
                event_id,
                home_team,
                away_team,
                league_id,
                league_name,
                starts,
                home_score,
                away_score,
                result
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """

        self.connect_db()
        inserted = 0
        with self.conn.cursor() as cursor:
            for _, row in merged.iterrows():
                cursor.execute(
                    insert_query,
                    (
                        row["event_id"],
                        row["home_team_x"],   # from odds1x2
                        row["away_team_x"],
                        row["league_id"],
                        row["league_name"],
                        row["starts"],
                        int(row["home_score_fb"]),  # from FBref
                        int(row["away_score_fb"]),
                        row["result"],
                    ),
                )
                inserted += 1

        self.conn.commit()
        self.logger.info(f"Inserted {inserted} results into results1 table.")
        self.close_db()


if __name__ == "__main__":
    DATABASE_URL = os.environ["DATABASE_URL"]
    SEASON = "2025"
    LEAGUE = "ENG-Premier League"  # or ENG-Premier League etc.

    updater = ResultsUpdaterSoccerdata(
        database_url=DATABASE_URL,
        league=LEAGUE,
        season=SEASON,
    )
    updater.update_results()
