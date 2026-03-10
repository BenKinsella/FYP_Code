from bs4 import BeautifulSoup
import psycopg2
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
import os

class OddsPortalScraper:
    def __init__(self, database_url, filepath="C:/Users/kinse/Downloads/Premier League results & scores, Football England _ OddsPortal(5).html"):
        self.filepath = filepath
        self.database_url = database_url
        self.conn = None

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("OddsPortalScraper")

    def load_page(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            return f.read()
        
    # ---------- DB helpers ----------
    def connect_db(self):
        if self.conn is None:
            self.conn = psycopg2.connect(self.database_url)

    def close_db(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def parse_matches(self, html):
        soup = BeautifulSoup(html, "html.parser")
        results = []
        current_date = None

        for match in soup.find_all("div", class_="eventRow"):
            # date-header only appears on the FIRST match of each date group
            # carry it forward for subsequent matches on the same date
            date_el = match.find(attrs={"data-testid": "date-header"})
            if date_el:
                try:
                    current_date = pd.to_datetime(
                        date_el.get_text(strip=True), format="%d %b %Y"
                    ).replace(tzinfo=timezone.utc)
                except ValueError as e:
                    self.logger.warning(f"Date parse failed: {date_el.get_text(strip=True)} - {e}")

            teams = match.find_all("p", class_="participant-name")
            scores = match.find_all("div", class_="min-mt:!hidden")

            if len(teams) == 2 and len(scores) == 2:
                results.append({
                    "home_team": teams[0].get_text(strip=True),
                    "away_team": teams[1].get_text(strip=True),
                    "home_score": scores[0].get_text(strip=True),
                    "away_score": scores[1].get_text(strip=True),
                    "starts": current_date,
                })

        self.logger.info(f"Parsed {len(results)} matches.")
        return results

    
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
    
    def get_result(self, home_score, away_score):
        if home_score > away_score:
            return "home_win"
        elif home_score < away_score:
            return "away_win"
        else:
            return "draw"

    '''
    def save_to_csv(self, results, filename="matches.csv"):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["home_team", "away_team", "score_home", "score_away", "1", "X", "2"],
                delimiter=",",   # use , so Excel treats headers properly in many locales
                quoting=csv.QUOTE_MINIMAL
            )
            writer.writeheader()
            writer.writerows(results)
    '''

    def update_results(self):
        odds_df = self.fetch_completed_odds_events()
        ## results_df = self.fetch_fbref_results()
        ## results_df = self.fetch_fotmob_results()
        ## results_df = self.fetch_matchhistory_results()
        results_list = self.parse_matches(self.load_page())
        results_df = pd.DataFrame(results_list)

        # ----- PRINT ODDS PORTAL RESULTS -----
        print("\n=== ODDS PORTAL RESULTS ===")
        print(results_df[["home_team", "away_team", "home_score", "away_score", "starts"]].head(10))
        print(f"Date range: {results_df['starts'].min()} to {results_df['starts'].max()}")
        print("============================\n")

        # Normalise names to `_lc` like odds_df
        results_df["home_team_lc"] = results_df["home_team"].astype(str).str.strip().str.lower()
        results_df["away_team_lc"] = results_df["away_team"].astype(str).str.strip().str.lower()

        # Use the parsed timestamp for date_only
        results_df["date_only"] = pd.to_datetime(results_df["starts"]).dt.date

        # For soft-match section that expects *_fbref columns:
        results_df["home_team_lc_fbref"] = results_df["home_team_lc"]
        results_df["away_team_lc_fbref"] = results_df["away_team_lc"]

        self.logger.info(f"Fetched {len(odds_df)} completed Pinnacle matches.")
        self.logger.info(f"Fetched {len(results_df)} matches from OddsPortal.")

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
            suffixes=("_odds", "_res")
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
            suffixes=("_odds", "_res")
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
                suffixes=("_odds", "_res"),
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
                lambda r: name_soft_match(r["home_team_lc_odds"], r["home_team_lc_res"])
                or name_soft_match(r["away_team_lc_odds"], r["away_team_lc_res"]),
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
        merged = merged.dropna(subset=["home_score", "away_score"])
        merged["home_score"] = merged["home_score"].astype(int)
        merged["away_score"] = merged["away_score"].astype(int)

        merged["result"] = merged.apply(
            lambda r: self.get_result(r["home_score"], r["away_score"]), axis=1
        )

        def coalesce_col(df, *col_names):
            for col in col_names:
                if col in df.columns:
                    return df[col]
            raise KeyError(f"None of {col_names} found in DataFrame. Available: {list(df.columns)}")

        merged["home_team"]   = coalesce_col(merged, "home_team_odds",   "home_team_x",   "home_team_res")
        merged["away_team"]   = coalesce_col(merged, "away_team_odds",   "away_team_x",   "away_team_res")
        merged["starts"]      = coalesce_col(merged, "starts_odds",      "starts_x",      "starts")
        merged["league_id"]   = coalesce_col(merged, "league_id_odds",   "league_id_x",   "league_id")
        merged["league_name"] = coalesce_col(merged, "league_name_odds", "league_name_x", "league_name")

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
                        row["home_team"],   # from odds1x2
                        row["away_team"],
                        row["league_id"],
                        row["league_name"],
                        row["starts"],
                        int(row["home_score"]),  # from FBref
                        int(row["away_score"]),
                        row["result"],
                    ),
                )
                inserted += 1

        self.conn.commit()
        self.logger.info(f"Inserted {inserted} results into results1 table.")
        self.close_db()

if __name__ == "__main__":
    DATABASE_URL = os.environ["DATABASE_URL"]

    # scraper = OddsPortalScraper()
    # matches = scraper.parse_matches(scraper.load_page())

    updater = OddsPortalScraper(
        database_url=DATABASE_URL
    )
    updater.update_results()
