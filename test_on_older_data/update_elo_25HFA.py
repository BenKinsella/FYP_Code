"""
Premier League Elo Updater — Dynamic HFA Variant
-------------------------------------------------
HFA is estimated dynamically using exponential decay weighting.
After each match day (not each match), the average implied HFA
across all matches played that day is used to update the estimate.

Switch between datasets by changing DATASET at the top:
  DATASET = 1  →  results1  / elo1_hfa   (your existing pipeline)
  DATASET = 2  →  results2  / elo2_hfa   (football-data.co.uk CSV)

For DATASET = 1, initial Elos are loaded from CSV_PATH.
For DATASET = 2, initial Elos are loaded from initial_elos2 in the DB.
"""

import psycopg2
import os
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Dataset switch ────────────────────────────────────────────────────────────
DATASET = 2   # change to 2 to run on football-data.co.uk data

# ── Config ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ["DATABASE_URL"]
LEAGUE_ID  = 1980
INIT_ELO   = 1500
HFA_INIT   = 42.0
ALPHA      = 0.10
K          = 20

# Dataset 1 only
CSV_PATH   = "C:/Users/kinse/Downloads/2025-10-08.csv"
START_DATE_1 = date(2025, 10, 8)

# Dataset 2 — start date is first match in football-data CSV
START_DATE_2 = date(2024, 8, 16)

# ── Derived from DATASET ──────────────────────────────────────────────────────
RESULTS_TABLE = "results1"  if DATASET == 1 else "results2"
ELO_TABLE     = "elo1_hfa"  if DATASET == 1 else "elo2_hfa"
START_DATE    = START_DATE_1 if DATASET == 1 else START_DATE_2

# ── PL team whitelist ─────────────────────────────────────────────────────────
PL_TEAMS = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton", "Chelsea", "Crystal Palace", "Everton",
    "Fulham", "Ipswich Town", "Leicester City", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Southampton", "Tottenham Hotspur",
    "West Ham United", "Wolverhampton",
}

# ── CSV → results1 name mapping (dataset 1 only) ──────────────────────────────
CSV_TO_RESULTS: dict[str, str] = {
    "Man City":    "Manchester City",
    "Man United":  "Manchester United",
    "Newcastle":   "Newcastle United",
    "Forest":      "Nottingham Forest",
    "Tottenham":   "Tottenham Hotspur",
    "West Ham":    "West Ham United",
    "Wolves":      "Wolverhampton",
    "Leicester":   "Leicester City",
    "Ipswich":     "Ipswich Town",
}

def csv_to_results1(name: str) -> str:
    return CSV_TO_RESULTS.get(name, name)


# ── Elo engine with dynamic HFA ───────────────────────────────────────────────
class FootballEloDynamicHFA:
    def __init__(self, initial_elos: dict, init_elo=INIT_ELO,
                 hfa_init=HFA_INIT, alpha=ALPHA, k=K):
        self.elo         = initial_elos.copy()
        self.init_elo    = init_elo
        self.hfa         = float(hfa_init)
        self.alpha       = alpha
        self.k           = k
        self.hfa_history = []

    def expected(self, elo_home: float, elo_away: float, hfa: float) -> float:
        return 1 / (1 + 10 ** ((elo_away + hfa - elo_home) / 400))

    def implied_hfa(self, elo_home: float, elo_away: float,
                    home_goals: int, away_goals: int) -> float:
        S = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
        S_clipped = float(np.clip(S, 0.01, 0.99))
        return float((elo_home - elo_away) - 400 * np.log10(1 / S_clipped - 1))

    def update_elos(self, home_team: str, away_team: str,
                    home_goals: int, away_goals: int) -> float:
        elo_home  = self.elo.get(home_team, self.init_elo)
        elo_away  = self.elo.get(away_team, self.init_elo)
        E_home    = self.expected(elo_home, elo_away, self.hfa)
        S_home    = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
        goal_mult = float(np.sqrt(abs(home_goals - away_goals) + 1))
        delta     = self.k * goal_mult * (S_home - E_home)
        self.elo[home_team] = elo_home + delta
        self.elo[away_team] = elo_away - delta
        return self.implied_hfa(elo_home, elo_away, home_goals, away_goals)

    def update_hfa(self, implied_hfa_values: list, match_date: date) -> None:
        day_mean_hfa = float(np.mean(implied_hfa_values))
        self.hfa = float((1 - self.alpha) * self.hfa + self.alpha * day_mean_hfa)
        self.hfa_history.append({
            "date":          match_date,
            "hfa":           round(self.hfa, 3),
            "day_mean_impl": round(day_mean_hfa, 3),
            "n_matches":     len(implied_hfa_values),
        })

    def snapshot(self, teams: set) -> dict:
        return {t: round(self.elo.get(t, self.init_elo)) for t in teams}


# ── DB helpers ────────────────────────────────────────────────────────────────
def fetch_initial_elos_from_db(conn) -> dict:
    """Load starting Elos from initial_elos2 table (dataset 2 only)."""
    df = pd.read_sql_query("SELECT team, elo FROM initial_elos2", conn)
    return {row.team: float(row.elo) for row in df.itertuples()}


def fetch_results(conn) -> pd.DataFrame:
    query = f"""
        SELECT event_id, home_team, away_team,
               home_score, away_score,
               starts::date AS match_date
        FROM {RESULTS_TABLE}
        WHERE league_id     = %s
          AND starts::date >= %s
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY starts ASC
    """
    df = pd.read_sql_query(query, conn, params=(LEAGUE_ID, START_DATE))
    return df.drop_duplicates(subset=["event_id"])


def ensure_elo_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {ELO_TABLE} (
                team  VARCHAR(100),
                elo   INTEGER,
                date  DATE,
                hfa   NUMERIC(7,3)
            )
        """)
    conn.commit()


def clear_and_insert(conn, rows: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {ELO_TABLE}")
        psycopg2.extras.execute_batch(
            cur,
            f"INSERT INTO {ELO_TABLE} (team, elo, date, hfa) VALUES (%s, %s, %s, %s)",
            [(r["team"], r["elo"], r["date"], float(r["hfa"])) for r in rows],
            page_size=500,
        )
    conn.commit()
    print(f"Inserted {len(rows)} rows into {ELO_TABLE}.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Running dataset {DATASET} → {RESULTS_TABLE} / {ELO_TABLE}")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        # 1. Load initial Elos
        if DATASET == 1:
            elo_df = pd.read_csv(CSV_PATH)
            initial_elos = {
                csv_to_results1(club): float(elo)
                for club, elo in zip(elo_df["Club"], elo_df["Elo"])
            }
            print(f"Loaded {len(initial_elos)} teams from CSV.")
        else:
            initial_elos = fetch_initial_elos_from_db(conn)
            print(f"Loaded {len(initial_elos)} teams from initial_elos2.")

        missing = PL_TEAMS - set(initial_elos.keys())
        if missing:
            print(f"WARNING: missing teams (will use {INIT_ELO}): {sorted(missing)}")

        engine = FootballEloDynamicHFA(initial_elos)

        ensure_elo_table(conn)

        # 2. Fetch results
        results = fetch_results(conn)
        print(f"Fetched {len(results)} completed PL matches from {RESULTS_TABLE}.")

        if results.empty:
            print("No results found.")
            return

        results["match_date"] = pd.to_datetime(results["match_date"]).dt.date
        matches_by_date = results.groupby("match_date")
        end_date        = date.today() if DATASET == 1 else results["match_date"].max()

        rows_to_insert: list[dict] = []
        current_date = START_DATE

        while current_date <= end_date:
            if current_date in matches_by_date.groups:
                day_matches = matches_by_date.get_group(current_date)

                implied_hfas = []
                for _, match in day_matches.iterrows():
                    impl = engine.update_elos(
                        home_team  = match["home_team"],
                        away_team  = match["away_team"],
                        home_goals = int(match["home_score"]),
                        away_goals = int(match["away_score"]),
                    )
                    implied_hfas.append(impl)

                engine.update_hfa(implied_hfas, current_date)
                print(f"  {current_date}: {len(day_matches)} match(es). "
                      f"Day implied HFA={np.mean(implied_hfas):.1f} → "
                      f"Running HFA={engine.hfa:.2f}")

            snap = engine.snapshot(PL_TEAMS)
            for team, elo in snap.items():
                rows_to_insert.append({
                    "team": team,
                    "elo":  elo,
                    "date": current_date,
                    "hfa":  float(engine.hfa),
                })

            current_date += timedelta(days=1)

        # 3. Write to DB
        print(f"\nWriting {len(rows_to_insert)} rows to {ELO_TABLE}...")
        clear_and_insert(conn, rows_to_insert)

        # 4. HFA evolution summary
        hfa_df = pd.DataFrame(engine.hfa_history)
        print(f"\n── HFA Evolution (start={HFA_INIT}, alpha={ALPHA}) ──")
        print(f"  Starting HFA : {HFA_INIT}")
        print(f"  Final HFA    : {hfa_df['hfa'].iloc[-1]}")
        print(f"  Min / Max    : {hfa_df['hfa'].min()} / {hfa_df['hfa'].max()}")
        print(f"  Mean implied : {hfa_df['day_mean_impl'].mean():.2f}")
        print(f"\nDone! {ELO_TABLE} is up to date.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()