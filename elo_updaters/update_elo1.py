import psycopg2
import os
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH     = "C:/Users/kinse/Downloads/2025-10-08.csv"
DATABASE_URL = os.environ["DATABASE_URL"]
LEAGUE_ID  = 1980
START_DATE = date(2025, 10, 8)
INIT_ELO   = 1500
HFA        = 39.5
K          = 20

# ── PL team whitelist (results1 names) ───────────────────────────────────────
# These are the exact names as they appear in results1.
# Only these 20 teams will ever be written to elo1.
PL_TEAMS = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton", "Chelsea", "Crystal Palace", "Everton",
    "Fulham", "Burnley", "Leeds United", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
    "West Ham United", "Wolverhampton",
}

# ── Team name mapping (CSV name → results1 name) ─────────────────────────────
# Used only when loading initial Elos from the CSV, so that the Elo engine
# keys match results1 names from the start.
CSV_TO_RESULTS: dict[str, str] = {
    "Man City":    "Manchester City",
    "Man United":  "Manchester United",
    "Newcastle":   "Newcastle United",
    "Forest":      "Nottingham Forest",
    "Tottenham":   "Tottenham Hotspur",
    "West Ham":    "West Ham United",
    "Wolves":      "Wolverhampton",
    "Leeds":       "Leeds United",
    # Add more here if the runtime warning flags others
}

def csv_to_results1(name: str) -> str:
    """Translate a CSV team name to its results1 equivalent."""
    return CSV_TO_RESULTS.get(name, name)


# ── Elo engine ────────────────────────────────────────────────────────────────
class FootballElo:
    def __init__(self, initial_elos: dict, init_elo=INIT_ELO, hfa=HFA, k=K):
        self.elo      = initial_elos.copy()
        self.init_elo = init_elo
        self.hfa      = hfa
        self.k        = k

    def expected(self, elo_home: float, elo_away: float) -> float:
        return 1 / (1 + 10 ** ((elo_away + self.hfa - elo_home) / 400))

    def update(self, home_team: str, away_team: str,
               home_goals: int, away_goals: int) -> tuple:
        elo_home = self.elo.get(home_team, self.init_elo)
        elo_away = self.elo.get(away_team, self.init_elo)

        E_home    = self.expected(elo_home, elo_away)
        S_home    = 1 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0)
        goal_mult = np.sqrt(abs(home_goals - away_goals) + 1)
        delta     = self.k * goal_mult * (S_home - E_home)

        self.elo[home_team] = elo_home + delta
        self.elo[away_team] = elo_away - delta
        return delta, self.elo[home_team], self.elo[away_team]

    def snapshot(self, teams: set) -> dict:
        """Return rounded Elos for the specified teams only."""
        return {t: round(self.elo.get(t, self.init_elo)) for t in teams}


# ── DB helpers ────────────────────────────────────────────────────────────────
def fetch_results(conn, league_id: int, from_date: date) -> pd.DataFrame:
    query = """
        SELECT
            event_id,
            home_team,
            away_team,
            home_score,
            away_score,
            starts::date AS match_date
        FROM results1
        WHERE league_id     = %s
          AND starts::date >= %s
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY starts ASC
    """
    df = pd.read_sql_query(query, conn, params=(league_id, from_date))
    return df.drop_duplicates(subset=["event_id"])


def clear_and_insert(conn, rows: list[dict]) -> None:
    """Wipe elo1 then bulk insert all rows from scratch."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM elo1")
        print("Cleared elo1 table.")
        psycopg2.extras.execute_batch(
            cur,
            "INSERT INTO elo1 (team, elo, date) VALUES (%s, %s, %s)",
            [(r["team"], r["elo"], r["date"]) for r in rows],
            page_size=500,
        )
    conn.commit()
    print(f"Inserted {len(rows)} rows into elo1.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load initial Elos from CSV, translating names to results1 format
    elo_df = pd.read_csv(CSV_PATH)
    initial_elos = {
        csv_to_results1(club): float(elo)
        for club, elo in zip(elo_df["Club"], elo_df["Elo"])
    }
    print(f"Loaded {len(initial_elos)} teams from CSV (stored under results1 names).")

    # Warn if any PL team is missing from the CSV after translation
    missing = PL_TEAMS - set(initial_elos.keys())
    if missing:
        print(f"WARNING: these PL teams are missing from the CSV (will use {INIT_ELO}):")
        for t in sorted(missing):
            print(f"  - '{t}'")

    engine = FootballElo(initial_elos)

    conn = psycopg2.connect(DATABASE_URL)
    try:
        # 2. Fetch results — names are already in results1 format, no mapping needed
        results = fetch_results(conn, LEAGUE_ID, START_DATE)
        print(f"Fetched {len(results)} completed PL matches from DB.")

        if results.empty:
            print("No results found – nothing to process.")
            return

        # Warn about any team in results that isn't in the whitelist
        all_result_teams = set(results["home_team"]) | set(results["away_team"])
        unexpected = all_result_teams - PL_TEAMS
        if unexpected:
            print("\nWARNING: teams in results1 not in PL_TEAMS:")
            for t in sorted(unexpected):
                print(f"  - '{t}'")
            print()

        # 3. Walk every calendar day and process matches
        results["match_date"] = pd.to_datetime(results["match_date"]).dt.date
        matches_by_date       = results.groupby("match_date")
        last_match_date       = max(matches_by_date.groups.keys())
        end_date              = date.today()

        rows_to_insert: list[dict] = []
        current_date = START_DATE

        while current_date <= end_date:
            if current_date in matches_by_date.groups:
                day_matches = matches_by_date.get_group(current_date)
                for _, match in day_matches.iterrows():
                    engine.update(
                        home_team  = match["home_team"],
                        away_team  = match["away_team"],
                        home_goals = int(match["home_score"]),
                        away_goals = int(match["away_score"]),
                    )
                print(f"  {current_date}: processed {len(day_matches)} match(es).")

            # Snapshot only the 20 PL teams (stored under results1 names)
            snap = engine.snapshot(PL_TEAMS)
            for team, elo in snap.items():
                rows_to_insert.append({"team": team, "elo": elo, "date": current_date})

            current_date += timedelta(days=1)

        # 4. Clear old data and write fresh
        print(f"\nWriting {len(rows_to_insert)} rows to elo1 "
              f"({len(PL_TEAMS)} teams × {(end_date - START_DATE).days + 1} days)...")
        clear_and_insert(conn, rows_to_insert)
        print("Done! elo1 is up to date.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()