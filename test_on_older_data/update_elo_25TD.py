"""
Premier League Elo Updater — Time-Decay Variant
------------------------------------------------
Identical to football_elo_updater.py with one change:
the K-factor is multiplied by an exponential time-decay weight
based on how many days have passed since each team's last match.

Based on: Hvattum & Arntzen (2010), "Using ELO ratings for match
result prediction in association football", International Journal
of Forecasting, 26(3), 460-470.

Switch between datasets by changing DATASET at the top:
  DATASET = 1  →  results1 / elo1_decay   (your existing pipeline)
  DATASET = 2  →  results2 / elo2_decay   (football-data.co.uk CSV)

For DATASET = 1, initial Elos are loaded from CSV_PATH.
For DATASET = 2, initial Elos are loaded from initial_elos2 in the DB.
decay_rate is calibrated separately per dataset.
"""

import psycopg2
import os
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.optimize import minimize_scalar

# ── Dataset switch ────────────────────────────────────────────────────────────
DATASET = 2   # change to 2 to run on football-data.co.uk data

# ── Config ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ["DATABASE_URL"]
LEAGUE_ID  = 1980
INIT_ELO   = 1500
HFA        = 42
K          = 20
DECAY_RATE = 0.004  # overwritten by calibration

# Dataset 1 only
CSV_PATH     = "C:/Users/kinse/Downloads/2025-10-08.csv"
START_DATE_1 = date(2025, 10, 8)

# Dataset 2 — first match in football-data CSV
START_DATE_2 = date(2024, 8, 16)

# ── Derived from DATASET ──────────────────────────────────────────────────────
RESULTS_TABLE = "results1"    if DATASET == 1 else "results2"
ELO_TABLE     = "elo1_decay"  if DATASET == 1 else "elo2_decay"
START_DATE    = START_DATE_1  if DATASET == 1 else START_DATE_2

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


# ── Elo engine with time-decay ────────────────────────────────────────────────
class FootballEloDecay:
    def __init__(self, initial_elos: dict, init_elo=INIT_ELO,
                 hfa=HFA, k=K, decay_rate=DECAY_RATE):
        self.elo        = initial_elos.copy()
        self.init_elo   = init_elo
        self.hfa        = hfa
        self.k          = k
        self.decay_rate = decay_rate
        self.last_played: dict[str, date] = {}

    def expected(self, elo_home: float, elo_away: float) -> float:
        return 1 / (1 + 10 ** ((elo_away + self.hfa - elo_home) / 400))

    def time_weight(self, team: str, match_date: date) -> float:
        last = self.last_played.get(team)
        if last is None:
            return 1.0
        days = (match_date - last).days
        return float(np.exp(-self.decay_rate * days))

    def update(self, home_team: str, away_team: str,
               home_goals: int, away_goals: int,
               match_date: date) -> tuple:
        elo_home  = self.elo.get(home_team, self.init_elo)
        elo_away  = self.elo.get(away_team, self.init_elo)
        E_home    = self.expected(elo_home, elo_away)
        S_home    = 1 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0)
        goal_mult = np.sqrt(abs(home_goals - away_goals) + 1)
        w_home    = self.time_weight(home_team, match_date)
        w_away    = self.time_weight(away_team, match_date)
        w         = (w_home + w_away) / 2
        delta     = self.k * w * goal_mult * (S_home - E_home)
        self.elo[home_team] = elo_home + delta
        self.elo[away_team] = elo_away - delta
        self.last_played[home_team] = match_date
        self.last_played[away_team] = match_date
        return delta, self.elo[home_team], self.elo[away_team]

    def snapshot(self, teams: set) -> dict:
        return {t: round(self.elo.get(t, self.init_elo)) for t in teams}


# ── Calibration ───────────────────────────────────────────────────────────────
def calibrate_decay_rate(results_df: pd.DataFrame,
                         initial_elos: dict,
                         draw_base: float,
                         draw_slope: float) -> float:
    def log_loss(decay_rate: float) -> float:
        engine = FootballEloDecay(initial_elos, decay_rate=decay_rate)
        results_df["match_date"] = pd.to_datetime(results_df["match_date"]).dt.date
        total_loss, n = 0.0, 0

        for _, row in results_df.sort_values("match_date").iterrows():
            elo_home = engine.elo.get(row["home_team"], INIT_ELO)
            elo_away = engine.elo.get(row["away_team"], INIT_ELO)
            E_home   = engine.expected(elo_home, elo_away)

            elo_diff = abs(elo_home - elo_away)
            p_draw   = max(0.01, draw_base - draw_slope * elo_diff / 400)
            p_home   = max(0.01, E_home - p_draw / 2)
            p_away   = max(0.01, 1 - E_home - p_draw / 2)
            total    = p_home + p_draw + p_away
            p_home  /= total; p_draw /= total; p_away /= total

            result   = str(row["result"]).strip().lower()
            p_actual = {"home_win": p_home, "draw": p_draw, "away_win": p_away}.get(result)
            if p_actual:
                total_loss -= np.log(max(p_actual, 1e-9))
                n += 1

            engine.update(
                row["home_team"], row["away_team"],
                int(row["home_score"]), int(row["away_score"]),
                row["match_date"],
            )

        return total_loss / n if n > 0 else 1e9

    result = minimize_scalar(log_loss, bounds=(0.001, 0.02), method="bounded")
    print(f"Calibrated decay_rate={result.x:.5f} (log-loss={result.fun:.4f})")
    return float(result.x)


# ── DB helpers ────────────────────────────────────────────────────────────────
def fetch_initial_elos_from_db(conn) -> dict:
    df = pd.read_sql_query("SELECT team, elo FROM initial_elos2", conn)
    return {row.team: float(row.elo) for row in df.itertuples()}


def fetch_results(conn) -> pd.DataFrame:
    query = f"""
        SELECT event_id, home_team, away_team,
               home_score, away_score, result,
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


def fetch_draw_params(conn) -> tuple:
    table = "match_odds_analysis_base" if DATASET == 1 else "match_odds_analysis2_base"
    row = pd.read_sql_query(
        f"SELECT draw_base, draw_slope FROM {table} LIMIT 1", conn
    ).iloc[0]
    return float(row["draw_base"]), float(row["draw_slope"])


def ensure_elo_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {ELO_TABLE} (
                team  VARCHAR(100),
                elo   INTEGER,
                date  DATE
            )
        """)
    conn.commit()


def clear_and_insert(conn, rows: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {ELO_TABLE}")
        print(f"Cleared {ELO_TABLE} table.")
        psycopg2.extras.execute_batch(
            cur,
            f"INSERT INTO {ELO_TABLE} (team, elo, date) VALUES (%s, %s, %s)",
            [(r["team"], r["elo"], r["date"]) for r in rows],
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

        ensure_elo_table(conn)

        # 2. Fetch results
        results = fetch_results(conn)
        print(f"Fetched {len(results)} completed PL matches from {RESULTS_TABLE}.")

        if results.empty:
            print("No results found.")
            return

        results["match_date"] = pd.to_datetime(results["match_date"]).dt.date

        # 3. Fetch shared draw params
        draw_base, draw_slope = fetch_draw_params(conn)
        print(f"Using draw_base={draw_base}, draw_slope={draw_slope}")

        # 4. Calibrate decay_rate
        print("\nCalibrating decay_rate...")
        decay_rate = calibrate_decay_rate(results.copy(), initial_elos, draw_base, draw_slope)

        # 5. Run engine with calibrated decay_rate
        engine          = FootballEloDecay(initial_elos, decay_rate=decay_rate)
        matches_by_date = results.groupby("match_date")
        end_date        = date.today() if DATASET == 1 else results["match_date"].max()

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
                        match_date = current_date,
                    )
                print(f"  {current_date}: processed {len(day_matches)} match(es).")

            snap = engine.snapshot(PL_TEAMS)
            for team, elo in snap.items():
                rows_to_insert.append({"team": team, "elo": elo, "date": current_date})

            current_date += timedelta(days=1)

        # 6. Write to DB
        print(f"\nWriting {len(rows_to_insert)} rows to {ELO_TABLE}...")
        clear_and_insert(conn, rows_to_insert)
        print(f"Done! decay_rate={decay_rate:.5f}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()