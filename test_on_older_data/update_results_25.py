"""
Football-Data.co.uk CSV Loader
-------------------------------
Loads a football-data.co.uk Premier League CSV (e.g. E0.csv) into:

  results2  — match results, same schema as results1
  odds2     — Pinnacle open + close odds, two rows per match
              (open: match_date 10:00, close: match_date 11:00)
              same schema as odds1x2

Set INITIAL_ELOS below using results1-style team names before running.
Any team not listed defaults to INIT_ELO (1500).
"""

import psycopg2
import os
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH     = "C:/Users/kinse/Downloads/E0.csv"
DATABASE_URL = os.environ["DATABASE_URL"]
LEAGUE_ID   = 1980
LEAGUE_NAME = "England Premier League"
INIT_ELO    = 1500

# ── Starting Elos ─────────────────────────────────────────────────────────────
# Replace these values with your own from ClubElo or your elo1 table
# at the start of the 24/25 season. Uses results1-style team names.
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
    return FD_TO_RESULTS.get(name.strip(), name.strip())


# ── DB setup ──────────────────────────────────────────────────────────────────
def setup_tables(conn) -> None:
    with conn.cursor() as cur:
        # results2 — mirrors results1 exactly
        cur.execute("DROP TABLE IF EXISTS results2")
        cur.execute("""
            CREATE TABLE results2 (
                event_id    SERIAL PRIMARY KEY,
                home_team   VARCHAR(100),
                away_team   VARCHAR(100),
                league_id   INTEGER,
                league_name VARCHAR(100),
                starts      TIMESTAMP,
                home_score  INTEGER,
                away_score  INTEGER,
                result      VARCHAR(20)
            )
        """)
        # odds2 — mirrors odds1x2 schema exactly
        cur.execute("DROP TABLE IF EXISTS odds2")
        cur.execute("""
            CREATE TABLE odds2 (
                event_id        BIGINT,
                home_team       VARCHAR(100),
                away_team       VARCHAR(100),
                starts          TIMESTAMP,
                logged_time     TIMESTAMP,
                league_id       INTEGER,
                league_name     VARCHAR(100),
                home_odds       NUMERIC(7,3),
                draw_odds       NUMERIC(7,3),
                away_odds       NUMERIC(7,3)
            )
        """)
    conn.commit()
    print("Tables created: results2, odds2")


def load_data(conn, df: pd.DataFrame) -> None:
    results_rows  = []
    odds_rows     = []
    skipped       = 0

    for i, row in df.iterrows():
        home = to_results1(str(row["HomeTeam"]))
        away = to_results1(str(row["AwayTeam"]))

        # Parse date — football-data uses DD/MM/YYYY
        try:
            match_dt = pd.to_datetime(row["Date"], dayfirst=True)
        except Exception:
            print(f"  Row {i}: skipping — bad date: {row['Date']}")
            skipped += 1
            continue

        # Scores
        try:
            home_score = int(row["FTHG"])
            away_score = int(row["FTAG"])
        except Exception:
            print(f"  Row {i}: skipping — bad score")
            skipped += 1
            continue

        # Result
        result = {"H": "home_win", "D": "draw", "A": "away_win"}.get(str(row["FTR"]).strip())
        if result is None:
            print(f"  Row {i}: skipping — unknown FTR: {row['FTR']}")
            skipped += 1
            continue

        event_id = i + 1  # matches SERIAL order (1-indexed)

        results_rows.append((
            home, away, LEAGUE_ID, LEAGUE_NAME,
            match_dt, home_score, away_score, result
        ))

        # Pinnacle odds — PSH/PSD/PSA = opening, PSCH/PSCD/PSCA = closing
        # Insert as two rows so odds_analysis ROW_NUMBER() logic works unchanged:
        #   open  row: logged_time = match date 10:00
        #   close row: logged_time = match date 11:00
        try:
            open_h  = float(row["PSH"])
            open_d  = float(row["PSD"])
            open_a  = float(row["PSA"])
            close_h = float(row["PSCH"])
            close_d = float(row["PSCD"])
            close_a = float(row["PSCA"])
            open_time  = match_dt.replace(hour=10, minute=0)
            close_time = match_dt.replace(hour=11, minute=0)
            odds_rows.append((event_id, home, away, match_dt, open_time,  1980, LEAGUE_NAME, open_h,  open_d,  open_a))
            odds_rows.append((event_id, home, away, match_dt, close_time, 1980, LEAGUE_NAME, close_h, close_d, close_a))
        except Exception:
            print(f"  Row {i}: missing Pinnacle odds for {home} vs {away} — odds rows skipped")

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
    print(f"Inserted {len(results_rows)} rows into results2 ({skipped} skipped).")

    # Insert into odds2 — two rows per match (open + close)
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO odds2
                (event_id, home_team, away_team, starts, logged_time,
                 league_id, league_name, home_odds, draw_odds, away_odds)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            odds_rows,
            page_size=200,
        )
    print(f"Inserted {len(odds_rows)} rows into odds2 ({len(odds_rows)//2} matches × 2 open/close).")

    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    # Drop rows without a result (future fixtures)
    df = df[df["FTR"].isin(["H", "D", "A"])].reset_index(drop=True)
    print(f"Loaded {len(df)} completed matches from CSV.")

    # Show name mappings being applied
    all_teams = sorted(set(df["HomeTeam"].tolist()) | set(df["AwayTeam"].tolist()))
    for t in all_teams:
        mapped = to_results1(t)
        if mapped != t:
            print(f"  Name mapped: '{t}' → '{mapped}'")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        setup_tables(conn)
        load_data(conn, df)
        print("\nDone!")
        print(f"  results2 : {len(df)} matches")
        print(f"  odds2    : {len(df) * 2} rows added (open + close per match)")
        print(f"\nStarting Elos set in INITIAL_ELOS — {len(INITIAL_ELOS)} teams.")
        print("Update these values before running your elo updater scripts.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()