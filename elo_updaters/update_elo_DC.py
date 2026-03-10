"""
Premier League Elo Updater — Dixon-Coles Variant
-------------------------------------------------
Identical to football_elo_updater.py with one change:
the Elo update delta is multiplied by a Dixon-Coles correction
factor that adjusts for the noisiness of low-scoring results.

Based on: Dixon & Coles (1997), "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market",
Journal of the Royal Statistical Society, Series C, 46(2), 265-280.

The correction factor τ (tau) adjusts the K-factor weight for
four scorelines that are systematically over/under-represented
relative to what the model expects:

    0-0  →  weight multiplied by (1 + rho)   [draws over-represented]
    1-1  →  weight multiplied by (1 + rho)   [draws over-represented]
    1-0  →  weight multiplied by (1 - rho)   [narrow home wins noisy]
    0-1  →  weight multiplied by (1 - rho)   [narrow away wins noisy]
    else →  weight unchanged (multiplied by 1)

rho is calibrated by minimising log-loss over historical results,
using the same draw_base/draw_slope as the base model so that rho
is the only variable being compared.

Stores results in elo1_dc (same schema as elo1).
"""

import psycopg2
import os
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.optimize import minimize_scalar

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH     = "C:/Users/kinse/Downloads/2025-10-08.csv"
DATABASE_URL = os.environ["DATABASE_URL"]
LEAGUE_ID  = 1980
START_DATE = date(2025, 10, 8)
INIT_ELO   = 1500
HFA        = 42
K          = 20
RHO        = 0.1   # overwritten by calibration; typically 0.05-0.20

# ── PL team whitelist (results1 names) ───────────────────────────────────────
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
    return CSV_TO_RESULTS.get(name, name)


# ── Dixon-Coles correction ────────────────────────────────────────────────────
def dc_weight(home_goals: int, away_goals: int, rho: float) -> float:
    """
    Returns the Dixon-Coles correction multiplier for a given scoreline.
    Only the four low-scoring scorelines are adjusted; all others return 1.0.
    rho must be in (0, 1) to keep weights positive.
    """
    if   home_goals == 0 and away_goals == 0: return 1.0 + rho
    elif home_goals == 1 and away_goals == 1: return 1.0 + rho
    elif home_goals == 1 and away_goals == 0: return 1.0 - rho
    elif home_goals == 0 and away_goals == 1: return 1.0 - rho
    else:                                     return 1.0


# ── Elo engine with Dixon-Coles correction ────────────────────────────────────
class FootballEloDC:
    def __init__(self, initial_elos: dict, init_elo=INIT_ELO,
                 hfa=HFA, k=K, rho=RHO):
        self.elo      = initial_elos.copy()
        self.init_elo = init_elo
        self.hfa      = hfa
        self.k        = k
        self.rho      = rho

    def expected(self, elo_home: float, elo_away: float) -> float:
        return 1 / (1 + 10 ** ((elo_away + self.hfa - elo_home) / 400))

    def update(self, home_team: str, away_team: str,
               home_goals: int, away_goals: int) -> tuple:
        elo_home = self.elo.get(home_team, self.init_elo)
        elo_away = self.elo.get(away_team, self.init_elo)

        E_home    = self.expected(elo_home, elo_away)
        S_home    = 1 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0)
        goal_mult = np.sqrt(abs(home_goals - away_goals) + 1)

        # Dixon-Coles correction — only affects the four noisy low-score lines
        tau       = dc_weight(home_goals, away_goals, self.rho)

        delta = self.k * tau * goal_mult * (S_home - E_home)

        self.elo[home_team] = elo_home + delta
        self.elo[away_team] = elo_away - delta
        return delta, self.elo[home_team], self.elo[away_team]

    def snapshot(self, teams: set) -> dict:
        return {t: round(self.elo.get(t, self.init_elo)) for t in teams}


# ── Calibration ───────────────────────────────────────────────────────────────
def calibrate_rho(results_df: pd.DataFrame,
                  initial_elos: dict,
                  draw_base: float,
                  draw_slope: float) -> float:
    """
    Find the rho that minimises log-loss over all historical matches.
    Uses the same draw_base/draw_slope as the base model so rho is the
    only variable being optimised.
    """
    def log_loss(rho: float) -> float:
        engine = FootballEloDC(initial_elos, rho=rho)
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
            )

        return total_loss / n if n > 0 else 1e9

    result = minimize_scalar(log_loss, bounds=(0.001, 0.5), method="bounded")
    print(f"Calibrated rho={result.x:.5f} (log-loss={result.fun:.4f})")
    return float(result.x)


# ── DB helpers ────────────────────────────────────────────────────────────────
def fetch_results(conn, league_id: int, from_date: date) -> pd.DataFrame:
    query = """
        SELECT event_id, home_team, away_team,
               home_score, away_score, result,
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


def ensure_dc_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS elo1_dc (
                team  VARCHAR(100),
                elo   INTEGER,
                date  DATE
            )
        """)
    conn.commit()


def clear_and_insert(conn, rows: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM elo1_dc")
        print("Cleared elo1_dc table.")
        psycopg2.extras.execute_batch(
            cur,
            "INSERT INTO elo1_dc (team, elo, date) VALUES (%s, %s, %s)",
            [(r["team"], r["elo"], r["date"]) for r in rows],
            page_size=500,
        )
    conn.commit()
    print(f"Inserted {len(rows)} rows into elo1_dc.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load initial Elos from CSV
    elo_df = pd.read_csv(CSV_PATH)
    initial_elos = {
        csv_to_results1(club): float(elo)
        for club, elo in zip(elo_df["Club"], elo_df["Elo"])
    }
    print(f"Loaded {len(initial_elos)} teams from CSV.")

    missing = PL_TEAMS - set(initial_elos.keys())
    if missing:
        print(f"WARNING: missing from CSV (will use {INIT_ELO}): {sorted(missing)}")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        ensure_dc_table(conn)

        # 2. Fetch results
        results = fetch_results(conn, LEAGUE_ID, START_DATE)
        print(f"Fetched {len(results)} completed PL matches from DB.")

        if results.empty:
            print("No results found.")
            return

        results["match_date"] = pd.to_datetime(results["match_date"]).dt.date

        # 3. Fetch shared draw params from match_odds_analysis_base
        row = pd.read_sql_query(
            "SELECT draw_base, draw_slope FROM match_odds_analysis_base LIMIT 1", conn
        ).iloc[0]
        draw_base  = float(row["draw_base"])
        draw_slope = float(row["draw_slope"])
        print(f"Using shared draw_base={draw_base}, draw_slope={draw_slope}")

        # 4. Calibrate rho
        print("\nCalibrating rho...")
        rho = calibrate_rho(results.copy(), initial_elos, draw_base, draw_slope)

        # 5. Run the engine with calibrated rho
        engine = FootballEloDC(initial_elos, rho=rho)
        matches_by_date = results.groupby("match_date")
        end_date        = date.today()

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

            snap = engine.snapshot(PL_TEAMS)
            for team, elo in snap.items():
                rows_to_insert.append({"team": team, "elo": elo, "date": current_date})

            current_date += timedelta(days=1)

        # 6. Write to DB
        print(f"\nWriting {len(rows_to_insert)} rows to elo1_dc...")
        clear_and_insert(conn, rows_to_insert)
        print(f"Done! rho={rho:.5f}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()