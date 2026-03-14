"""
Premier League Elo Updater — Dynamic HFA Variant
-------------------------------------------------
HFA is estimated dynamically using ClubElo's method.
After each match day, HFA is updated based on whether home or
away teams won more Elo points that day:
    HFA += sum(delta) * 0.075

Stores results in elo1_hfa (same schema as elo1 + hfa column).
"""

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
HFA_INIT   = 39.5
HFA_RATE   = 0.005   # ClubElo's HFA update rate
K          = 20

# ── PL team whitelist (results1 names) ───────────────────────────────────────
PL_TEAMS = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton", "Chelsea", "Crystal Palace", "Everton",
    "Fulham", "Burnley", "Leeds United", "Liverpool",
    "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
    "West Ham United", "Wolverhampton",
}

# ── CSV → results1 name mapping ───────────────────────────────────────────────
CSV_TO_RESULTS: dict[str, str] = {
    "Man City":    "Manchester City",
    "Man United":  "Manchester United",
    "Newcastle":   "Newcastle United",
    "Forest":      "Nottingham Forest",
    "Tottenham":   "Tottenham Hotspur",
    "West Ham":    "West Ham United",
    "Wolves":      "Wolverhampton",
    "Leeds":       "Leeds United",
}

def csv_to_results1(name: str) -> str:
    return CSV_TO_RESULTS.get(name, name)


# ── Elo engine with dynamic HFA (ClubElo method) ─────────────────────────────
class FootballEloDynamicHFA:
    def __init__(self, initial_elos: dict, init_elo=INIT_ELO,
                 hfa_init=HFA_INIT, hfa_rate=HFA_RATE, k=K):
        self.elo         = initial_elos.copy()
        self.init_elo    = init_elo
        self.hfa         = float(hfa_init)
        self.hfa_rate    = hfa_rate
        self.k           = k
        self.hfa_history = []

    def expected(self, elo_home: float, elo_away: float, hfa: float) -> float:
        return 1 / (1 + 10 ** ((elo_away + hfa - elo_home) / 400))

    def update_elos(self, home_team: str, away_team: str,
                    home_goals: int, away_goals: int) -> float:
        """Apply Elo update using current HFA. Returns delta for HFA update."""
        elo_home = self.elo.get(home_team, self.init_elo)
        elo_away = self.elo.get(away_team, self.init_elo)

        E_home    = self.expected(elo_home, elo_away, self.hfa)
        S_home    = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
        goal_mult = float(np.sqrt(abs(home_goals - away_goals) + 1))
        delta     = self.k * goal_mult * (S_home - E_home)

        self.elo[home_team] = elo_home + delta
        self.elo[away_team] = elo_away - delta

        return delta

    def update_hfa(self, deltas: list, match_date: date) -> None:
        """
        Update HFA using ClubElo's method:
            HFA += sum(delta) * 0.075
        If home teams collectively gained more Elo points, HFA increases.
        If away teams gained more, HFA decreases.
        """
        delta_sum = float(np.sum(deltas))
        self.hfa  = float(self.hfa + delta_sum * self.hfa_rate)
        self.hfa_history.append({
            "date":      match_date,
            "hfa":       round(self.hfa, 3),
            "delta_sum": round(delta_sum, 3),
            "n_matches": len(deltas),
        })

    def snapshot(self, teams: set) -> dict:
        return {t: round(self.elo.get(t, self.init_elo)) for t in teams}


# ── DB helpers ────────────────────────────────────────────────────────────────
def fetch_results(conn, league_id: int, from_date: date) -> pd.DataFrame:
    query = """
        SELECT event_id, home_team, away_team,
               home_score, away_score,
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


def ensure_hfa_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS elo1_hfa (
                team  VARCHAR(100),
                elo   INTEGER,
                date  DATE,
                hfa   NUMERIC(7,3)
            )
        """)
    conn.commit()


def clear_and_insert(conn, rows: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM elo1_hfa")
        psycopg2.extras.execute_batch(
            cur,
            "INSERT INTO elo1_hfa (team, elo, date, hfa) VALUES (%s, %s, %s, %s)",
            [(r["team"], r["elo"], r["date"], float(r["hfa"])) for r in rows],
            page_size=500,
        )
    conn.commit()
    print(f"Inserted {len(rows)} rows into elo1_hfa.")


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

    engine = FootballEloDynamicHFA(initial_elos)

    conn = psycopg2.connect(DATABASE_URL)
    try:
        ensure_hfa_table(conn)

        # 2. Fetch results
        results = fetch_results(conn, LEAGUE_ID, START_DATE)
        print(f"Fetched {len(results)} completed PL matches from DB.")

        if results.empty:
            print("No results found.")
            return

        results["match_date"] = pd.to_datetime(results["match_date"]).dt.date
        matches_by_date       = results.groupby("match_date")
        end_date              = date.today()

        rows_to_insert: list[dict] = []
        current_date = START_DATE

        while current_date <= end_date:
            if current_date in matches_by_date.groups:
                day_matches = matches_by_date.get_group(current_date)

                # Step 1: update Elos for all matches today, collect deltas
                deltas = []
                for _, match in day_matches.iterrows():
                    delta = engine.update_elos(
                        home_team  = match["home_team"],
                        away_team  = match["away_team"],
                        home_goals = int(match["home_score"]),
                        away_goals = int(match["away_score"]),
                    )
                    deltas.append(delta)

                # Step 2: update HFA once using sum of today's deltas
                engine.update_hfa(deltas, current_date)
                print(f"  {current_date}: {len(day_matches)} match(es). "
                      f"Delta sum={np.sum(deltas):.1f} → "
                      f"Running HFA={engine.hfa:.2f}")

            # Snapshot PL teams with current HFA
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
        print(f"\nWriting {len(rows_to_insert)} rows to elo1_hfa...")
        clear_and_insert(conn, rows_to_insert)

        # 4. HFA evolution summary
        hfa_df = pd.DataFrame(engine.hfa_history)
        print(f"\n── HFA Evolution (start={HFA_INIT}, rate={HFA_RATE}) ──")
        print(f"  Starting HFA : {HFA_INIT}")
        print(f"  Final HFA    : {hfa_df['hfa'].iloc[-1]}")
        print(f"  Min / Max    : {hfa_df['hfa'].min()} / {hfa_df['hfa'].max()}")
        print(f"  Mean delta_sum: {hfa_df['delta_sum'].mean():.2f}")
        print("\nDone! elo1_hfa is up to date.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()