"""
PL Odds Analysis Pipeline
--------------------------
1. Calibrates draw_base and draw_slope against historical PL results
2. For each PL match computes:
     - Opening Elo odds  (Elo on the date of first Pinnacle entry)
     - Closing Elo odds  (Elo from the day before kick-off)
     - Opening Pinnacle implied probs (normalised)
     - Closing Pinnacle implied probs (normalised)
     - Information gain: -log2(p_actual) for Elo and Pinnacle at open & close
3. Upserts everything into match_odds_analysis
"""

import psycopg2
import os
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import date, timedelta
from scipy.optimize import minimize

# ── Config ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ["DATABASE_URL"]
LEAGUE_ID = 1980
HFA       = 42   # must match your elo updater


# ── Elo / odds helpers ────────────────────────────────────────────────────────
def expected_score(elo_home: float, elo_away: float, hfa: float = HFA) -> float:
    return 1 / (1 + 10 ** ((elo_away + hfa - elo_home) / 400))


def wdl_probs(elo_home: float, elo_away: float,
              draw_base: float, draw_slope: float) -> tuple:
    """Return (p_home, p_draw, p_away) using ClubElo-style draw model."""
    e_home    = expected_score(elo_home, elo_away)
    elo_diff  = abs(elo_home - elo_away)
    p_draw    = max(0.01, draw_base - draw_slope * elo_diff / 400)
    p_home    = max(0.01, e_home - p_draw / 2)
    p_away    = max(0.01, 1 - e_home - p_draw / 2)
    total     = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def normalise_pinnacle(home_odds: float, draw_odds: float, away_odds: float) -> tuple:
    """Convert decimal odds to normalised implied probabilities."""
    raw        = np.array([1 / home_odds, 1 / draw_odds, 1 / away_odds])
    normalised = raw / raw.sum()
    return float(normalised[0]), float(normalised[1]), float(normalised[2])


def information_gain(p: float) -> float:
    """Shannon information of the actual outcome: -log2(p)."""
    return float(-np.log2(max(p, 1e-9)))


# ── Calibration ───────────────────────────────────────────────────────────────
def calibrate_draw_params(matches_df: pd.DataFrame,
                          elo_lookup: dict) -> tuple[float, float]:
    """
    Fit draw_base and draw_slope by minimising log-loss over historical matches.
    Uses day-before-match Elo for each team.
    """
    def log_loss(params):
        draw_base, draw_slope = params
        if draw_base <= 0 or draw_slope < 0 or draw_base > 1:
            return 1e9
        total_loss = 0.0
        n = 0
        for _, row in matches_df.iterrows():
            prev_date = row["match_date"] - timedelta(days=1)
            eh = elo_lookup.get((row["home_team"], prev_date))
            ea = elo_lookup.get((row["away_team"], prev_date))
            if eh is None or ea is None:
                continue
            ph, pd_, pa = wdl_probs(eh, ea, draw_base, draw_slope)
            p_actual = {"home_win": ph, "draw": pd_, "away_win": pa}.get(row["result"])
            if p_actual is None:
                continue
            total_loss -= np.log(max(p_actual, 1e-9))
            n += 1
        return total_loss / n if n > 0 else 1e9

    result = minimize(
        log_loss,
        x0=[0.26, 0.20],
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000}
    )
    draw_base, draw_slope = result.x
    print(f"Calibrated: draw_base={draw_base:.4f}, draw_slope={draw_slope:.4f} "
          f"(log-loss={result.fun:.4f})")
    return float(draw_base), float(draw_slope)


# ── DB fetchers ───────────────────────────────────────────────────────────────
def fetch_elo_lookup(conn) -> dict:
    """Build {(team, date) -> elo} from elo1."""
    df = pd.read_sql_query("SELECT team, elo, date FROM elo1", conn)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return {(row.team, row.date): row.elo for row in df.itertuples()}


def fetch_matches(conn) -> pd.DataFrame:
    """Completed PL matches with result."""
    query = """
        SELECT
            event_id, home_team, away_team,
            starts, home_score, away_score, result,
            starts::date AS match_date
        FROM results1
        WHERE league_id = %s
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=(LEAGUE_ID,))
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    # Normalise result to lowercase (home_win, draw, away_win)
    df["result"] = df["result"].astype(str).str.strip().str.lower()
    return df.drop_duplicates(subset=["event_id"])


def fetch_pinnacle_open_close(conn) -> pd.DataFrame:
    """
    For each event_id get the opening row (earliest logged_time)
    and closing row (latest logged_time before starts).
    Returns one row per event with both open and close odds.
    """
    query = """
        WITH ranked AS (
            SELECT
                event_id,
                logged_time,
                home_odds,
                draw_odds,
                away_odds,
                ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY logged_time ASC)  AS rn_open,
                ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY logged_time DESC) AS rn_close
            FROM odds1x2
            WHERE league_id = %s
              AND home_odds IS NOT NULL
              AND draw_odds IS NOT NULL
              AND away_odds IS NOT NULL
        )
        SELECT
            event_id,
            MAX(CASE WHEN rn_open  = 1 THEN logged_time END) AS open_logged_time,
            MAX(CASE WHEN rn_open  = 1 THEN home_odds   END) AS open_home_odds,
            MAX(CASE WHEN rn_open  = 1 THEN draw_odds   END) AS open_draw_odds,
            MAX(CASE WHEN rn_open  = 1 THEN away_odds   END) AS open_away_odds,
            MAX(CASE WHEN rn_close = 1 THEN logged_time END) AS close_logged_time,
            MAX(CASE WHEN rn_close = 1 THEN home_odds   END) AS close_home_odds,
            MAX(CASE WHEN rn_close = 1 THEN draw_odds   END) AS close_draw_odds,
            MAX(CASE WHEN rn_close = 1 THEN away_odds   END) AS close_away_odds
        FROM ranked
        GROUP BY event_id
    """
    df = pd.read_sql_query(query, conn, params=(LEAGUE_ID,))
    df["open_date"]  = pd.to_datetime(df["open_logged_time"]).dt.date
    df["close_date"] = pd.to_datetime(df["close_logged_time"]).dt.date
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        # 1. Load data
        print("Fetching data...")
        elo_lookup = fetch_elo_lookup(conn)
        print(f"  Elo lookup: {len(elo_lookup)} (team, date) entries.")

        matches = fetch_matches(conn)
        print(f"  Matches: {len(matches)} completed PL matches.")

        pinnacle = fetch_pinnacle_open_close(conn)
        print(f"  Pinnacle: {len(pinnacle)} events with odds.")

        # 2. Calibrate draw params
        print("\nCalibrating draw parameters...")
        draw_base, draw_slope = calibrate_draw_params(matches, elo_lookup)

        # 3. Merge matches with pinnacle odds
        df = matches.merge(pinnacle, on="event_id", how="inner")
        print(f"\nBuilding analysis for {len(df)} matched events...")

        # 4. Compute everything row by row
        rows = []
        skipped = 0

        for _, row in df.iterrows():
            match_date = row["match_date"]
            prev_date  = match_date - timedelta(days=1)
            open_date  = row["open_date"]

            # Elo at opening: date of first Pinnacle entry
            elo_h_open = elo_lookup.get((row["home_team"], open_date))
            elo_a_open = elo_lookup.get((row["away_team"], open_date))

            # Elo at close: day before kick-off
            elo_h_close = elo_lookup.get((row["home_team"], prev_date))
            elo_a_close = elo_lookup.get((row["away_team"], prev_date))

            if None in (elo_h_open, elo_a_open, elo_h_close, elo_a_close):
                skipped += 1
                continue

            # Elo probabilities
            ph_o, pd_o, pa_o = wdl_probs(elo_h_open,  elo_a_open,  draw_base, draw_slope)
            ph_c, pd_c, pa_c = wdl_probs(elo_h_close, elo_a_close, draw_base, draw_slope)

            # Pinnacle normalised probabilities
            pin_ph_o, pin_pd_o, pin_pa_o = normalise_pinnacle(
                row["open_home_odds"], row["open_draw_odds"], row["open_away_odds"])
            pin_ph_c, pin_pd_c, pin_pa_c = normalise_pinnacle(
                row["close_home_odds"], row["close_draw_odds"], row["close_away_odds"])

            # Probability assigned to the actual outcome
            result      = row["result"]
            elo_p_open  = {"home_win": ph_o, "draw": pd_o, "away_win": pa_o}.get(result)
            elo_p_close = {"home_win": ph_c, "draw": pd_c, "away_win": pa_c}.get(result)
            pin_p_open  = {"home_win": pin_ph_o, "draw": pin_pd_o, "away_win": pin_pa_o}.get(result)
            pin_p_close = {"home_win": pin_ph_c, "draw": pin_pd_c, "away_win": pin_pa_c}.get(result)

            if None in (elo_p_open, elo_p_close, pin_p_open, pin_p_close):
                skipped += 1
                continue

            rows.append((
                int(row["event_id"]),
                str(row["home_team"]),
                str(row["away_team"]),
                row["starts"],
                int(row["home_score"]),
                int(row["away_score"]),
                result,
                int(elo_h_open),  int(elo_a_open),
                int(elo_h_close), int(elo_a_close),
                round(ph_o, 4), round(pd_o, 4), round(pa_o, 4),
                round(ph_c, 4), round(pd_c, 4), round(pa_c, 4),
                float(row["open_home_odds"]),  float(row["open_draw_odds"]),  float(row["open_away_odds"]),
                float(row["close_home_odds"]), float(row["close_draw_odds"]), float(row["close_away_odds"]),
                round(pin_ph_o, 4), round(pin_pd_o, 4), round(pin_pa_o, 4),
                round(pin_ph_c, 4), round(pin_pd_c, 4), round(pin_pa_c, 4),
                round(information_gain(elo_p_open),  4),
                round(information_gain(elo_p_close), 4),
                round(information_gain(pin_p_open),  4),
                round(information_gain(pin_p_close), 4),
                round(draw_base,  4),
                round(draw_slope, 4),
            ))

        print(f"  Computed: {len(rows)} rows | Skipped (missing Elo): {skipped}")

        # 5. Upsert into DB
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO match_odds_analysis (
                    event_id, home_team, away_team,
                    starts, home_score, away_score, result,
                    elo_home_open, elo_away_open, elo_home_close, elo_away_close,
                    elo_open_p_home,  elo_open_p_draw,  elo_open_p_away,
                    elo_close_p_home, elo_close_p_draw, elo_close_p_away,
                    pin_open_home_odds,  pin_open_draw_odds,  pin_open_away_odds,
                    pin_close_home_odds, pin_close_draw_odds, pin_close_away_odds,
                    pin_open_p_home,  pin_open_p_draw,  pin_open_p_away,
                    pin_close_p_home, pin_close_p_draw, pin_close_p_away,
                    ig_elo_open, ig_elo_close, ig_pin_open, ig_pin_close,
                    draw_base, draw_slope
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s
                )
                ON CONFLICT (event_id) DO UPDATE SET
                    home_team           = EXCLUDED.home_team,
                    away_team           = EXCLUDED.away_team,
                    starts              = EXCLUDED.starts,
                    home_score          = EXCLUDED.home_score,
                    away_score          = EXCLUDED.away_score,
                    result              = EXCLUDED.result,
                    elo_home_open       = EXCLUDED.elo_home_open,
                    elo_away_open       = EXCLUDED.elo_away_open,
                    elo_home_close      = EXCLUDED.elo_home_close,
                    elo_away_close      = EXCLUDED.elo_away_close,
                    elo_open_p_home     = EXCLUDED.elo_open_p_home,
                    elo_open_p_draw     = EXCLUDED.elo_open_p_draw,
                    elo_open_p_away     = EXCLUDED.elo_open_p_away,
                    elo_close_p_home    = EXCLUDED.elo_close_p_home,
                    elo_close_p_draw    = EXCLUDED.elo_close_p_draw,
                    elo_close_p_away    = EXCLUDED.elo_close_p_away,
                    pin_open_home_odds  = EXCLUDED.pin_open_home_odds,
                    pin_open_draw_odds  = EXCLUDED.pin_open_draw_odds,
                    pin_open_away_odds  = EXCLUDED.pin_open_away_odds,
                    pin_close_home_odds = EXCLUDED.pin_close_home_odds,
                    pin_close_draw_odds = EXCLUDED.pin_close_draw_odds,
                    pin_close_away_odds = EXCLUDED.pin_close_away_odds,
                    pin_open_p_home     = EXCLUDED.pin_open_p_home,
                    pin_open_p_draw     = EXCLUDED.pin_open_p_draw,
                    pin_open_p_away     = EXCLUDED.pin_open_p_away,
                    pin_close_p_home    = EXCLUDED.pin_close_p_home,
                    pin_close_p_draw    = EXCLUDED.pin_close_p_draw,
                    pin_close_p_away    = EXCLUDED.pin_close_p_away,
                    ig_elo_open         = EXCLUDED.ig_elo_open,
                    ig_elo_close        = EXCLUDED.ig_elo_close,
                    ig_pin_open         = EXCLUDED.ig_pin_open,
                    ig_pin_close        = EXCLUDED.ig_pin_close,
                    draw_base           = EXCLUDED.draw_base,
                    draw_slope          = EXCLUDED.draw_slope
                """,
                rows,
                page_size=200,
            )
        conn.commit()
        print(f"Done! {len(rows)} rows upserted into match_odds_analysis.")

        # 6. Print summary comparison
        summary = pd.read_sql_query("""
            SELECT
                ROUND(AVG(ig_elo_open),  4) AS avg_ig_elo_open,
                ROUND(AVG(ig_elo_close), 4) AS avg_ig_elo_close,
                ROUND(AVG(ig_pin_open),  4) AS avg_ig_pin_open,
                ROUND(AVG(ig_pin_close), 4) AS avg_ig_pin_close,
                COUNT(*) AS n_matches
            FROM match_odds_analysis
        """, conn)
        print("\n── Information Gain Summary (lower = better calibrated) ──")
        print(summary.to_string(index=False))

    finally:
        conn.close()


if __name__ == "__main__":
    main()