"""
Football-Data.co.uk CSV Loader
-------------------------------
Loads a football-data.co.uk Premier League CSV (e.g. E0.csv) into two DB tables:

  results2  — match results (same schema as results1)
  odds2     — Pinnacle opening + closing odds (normalised)

Usage:
  1. Set CSV_PATH to your downloaded file
  2. Set INITIAL_ELOS — a dict of {team_name: elo} using results1-style names.
     Any team not listed will default to INIT_ELO (1500).
  3. Run the script. It will DROP and recreate both tables on each run.

Team name mapping:
  football-data names are automatically translated to your results1-style names.
  Add entries to FD_TO_RESULTS if new names appear.
"""

import psycopg2
import os
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import date

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH     = "C:/Users/kinse/Downloads/E0.csv"
DATABASE_URL = os.environ["DATABASE_URL"]
LEAGUE_ID   = 1980
LEAGUE_NAME = "England Premier League"
INIT_ELO    = 1500

# ── Manual Elo inputs ─────────────────────────────────────────────────────────
# Set starting Elos here using results1-style team names.
# Teams not listed will use INIT_ELO.
# Example values — replace with your own from ClubElo or your elo1 table.
INITIAL_ELOS: dict[str, float] = {
    "Arsenal":               1950,
    "Aston Villa":           1773,
    "Bournemouth":           1691,
    "Brentford":             1715,
    "Brighton":              1725,
    "Chelsea":               1815,
    "Crystal Palace":        1750,
    "Everton":               1685,
    "Fulham":                1712,
    "Ipswich Town":          1580,
    "Leicester City":        1650,
    "Liverpool":             1918,
    "Manchester City":       2045,
    "Manchester United":     1780,
    "Newcastle United":      1805,
    "Nottingham Forest":     1650,
    "Southampton":           1585,
    "Tottenham Hotspur":     1790,
    "West Ham United":       1717,
    "Wolverhampton":         1675,
}

# ── Name mapping (football-data → results1) ───────────────────────────────────
FD_TO_RESULTS: dict[str, str] = {
    "Man City":      "Manchester City",
    "Man United":    "Manchester United",
    "Newcastle":     "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham":     "Tottenham Hotspur",
    "West Ham":      "West Ham United",
    "Wolves":        "Wolverhampton",
    "Leicester":     "Leicester City",
    "Ipswich":       "Ipswich Town",
}

def to_results1(name: str) -> str:
    return FD_TO_RESULTS.get(name, name)


# ── Odds helpers ──────────────────────────────────────────────────────────────
def normalise(h: float, d: float, a: float) -> tuple:
    """Normalise raw decimal odds to sum-to-1 probabilities."""
    try:
        raw = np.array([1/h, 1/d, 1/a])
        n   = raw / raw.sum()
        return float(n[0]), float(n[1]), float(n[2])
    except Exception:
        return None, None, None


# ── DB setup ──────────────────────────────────────────────────────────────────
def setup_tables(conn) -> None:
    with conn.cursor() as cur:
        # results2
        cur.execute("DROP TABLE IF EXISTS results2")
        cur.execute("""
            CREATE TABLE results2 (
                event_id     SERIAL PRIMARY KEY,
                home_team    VARCHAR(100),
                away_team    VARCHAR(100),
                league_id    INTEGER,
                league_name  VARCHAR(100),
                starts       DATE,
                home_score   INTEGER,
                away_score   INTEGER,
                result       VARCHAR(20)
            )
        """)

        # odds2 — normalised Pinnacle open + close probabilities
        cur.execute("DROP TABLE IF EXISTS odds2")
        cur.execute("""
            CREATE TABLE odds2 (
                event_id        INTEGER PRIMARY KEY,
                home_team       VARCHAR(100),
                away_team       VARCHAR(100),
                starts          DATE,
                pin_open_p_home NUMERIC(6,4),
                pin_open_p_draw NUMERIC(6,4),
                pin_open_p_away NUMERIC(6,4),
                pin_close_p_home NUMERIC(6,4),
                pin_close_p_draw NUMERIC(6,4),
                pin_close_p_away NUMERIC(6,4)
            )
        """)

        # elo2 — same schema as elo1, for storing decay/dc/base ratings on this dataset
        cur.execute("DROP TABLE IF EXISTS elo2")
        cur.execute("""
            CREATE TABLE elo2 (
                team  VARCHAR(100),
                elo   INTEGER,
                date  DATE
            )
        """)

        # initial_elos2 — stores the manually set starting Elos for reference
        cur.execute("DROP TABLE IF EXISTS initial_elos2")
        cur.execute("""
            CREATE TABLE initial_elos2 (
                team  VARCHAR(100) PRIMARY KEY,
                elo   NUMERIC(7,2)
            )
        """)

    conn.commit()
    print("Tables created: results2, odds2, elo2, initial_elos2")


def load_data(conn, df: pd.DataFrame) -> None:
    results_rows = []
    odds_rows    = []
    skipped_odds = 0

    for i, row in df.iterrows():
        home = to_results1(str(row["HomeTeam"]).strip())
        away = to_results1(str(row["AwayTeam"]).strip())

        # Parse date
        try:
            match_date = pd.to_datetime(row["Date"], dayfirst=True).date()
        except Exception:
            print(f"  Skipping row {i} — bad date: {row['Date']}")
            continue

        # Scores
        try:
            home_score = int(row["FTHG"])
            away_score = int(row["FTAG"])
        except Exception:
            print(f"  Skipping row {i} — bad score")
            continue

        # Result (H/D/A → home_win/draw/away_win)
        ftr = str(row["FTR"]).strip()
        result = {"H": "home_win", "D": "draw", "A": "away_win"}.get(ftr)
        if result is None:
            print(f"  Skipping row {i} — unknown FTR: {ftr}")
            continue

        event_id = i + 1  # 1-indexed, matches SERIAL order
        results_rows.append((
            home, away, LEAGUE_ID, LEAGUE_NAME,
            match_date, home_score, away_score, result
        ))

        # Pinnacle odds — open: PSH/PSD/PSA, close: PSCH/PSCD/PSCA
        try:
            open_ph,  open_pd,  open_pa  = normalise(float(row["PSH"]),  float(row["PSD"]),  float(row["PSA"]))
            close_ph, close_pd, close_pa = normalise(float(row["PSCH"]), float(row["PSCD"]), float(row["PSCA"]))
        except Exception:
            skipped_odds += 1
            open_ph = open_pd = open_pa = close_ph = close_pd = close_pa = None

        odds_rows.append((
            event_id, home, away, match_date,
            open_ph,  open_pd,  open_pa,
            close_ph, close_pd, close_pa,
        ))

    # Insert results2
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO results2
                (home_team, away_team, league_id, league_name,
                 starts, home_score, away_score, result)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            results_rows,
            page_size=200,
        )
    print(f"Inserted {len(results_rows)} rows into results2.")

    # Insert odds2
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO odds2
                (event_id, home_team, away_team, starts,
                 pin_open_p_home,  pin_open_p_draw,  pin_open_p_away,
                 pin_close_p_home, pin_close_p_draw, pin_close_p_away)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            odds_rows,
            page_size=200,
        )
    print(f"Inserted {len(odds_rows)} rows into odds2 ({skipped_odds} skipped — missing Pinnacle odds).")

    # Insert initial_elos2
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            "INSERT INTO initial_elos2 (team, elo) VALUES (%s,%s)",
            [(team, elo) for team, elo in INITIAL_ELOS.items()],
        )
    print(f"Inserted {len(INITIAL_ELOS)} rows into initial_elos2.")

    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    # Drop rows with no result (future fixtures in the file)
    df = df[df["FTR"].isin(["H", "D", "A"])].reset_index(drop=True)
    print(f"Loaded {len(df)} completed matches from {CSV_PATH}")

    # Warn about any unmapped team names
    all_teams = set(df["HomeTeam"].tolist()) | set(df["AwayTeam"].tolist())
    for t in sorted(all_teams):
        mapped = to_results1(t)
        if mapped != t:
            print(f"  Mapped: '{t}' → '{mapped}'")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        setup_tables(conn)
        load_data(conn, df)
        print("\nDone!")
        print("Tables ready: results2, odds2, elo2, initial_elos2")
        print("Next step: run your elo updater scripts pointing at results2/elo2,")
        print("then run odds_analysis pointing at odds2 instead of odds1x2.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()