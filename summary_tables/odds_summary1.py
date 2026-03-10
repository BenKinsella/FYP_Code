"""
PL Odds Analysis Pipeline
--------------------------
1. Calibrates draw_base / draw_slope against historical PL results
2. For each PL match computes odds and IG for four models:
     - Fixed-HFA Elo    (opening + closing)
     - Dynamic-HFA Elo  (opening + closing)
     - Pinnacle         (opening + closing)
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
HFA_FIXED = 42.0


# ── Elo / odds helpers ────────────────────────────────────────────────────────
def expected_score(elo_home: float, elo_away: float, hfa: float) -> float:
    return 1 / (1 + 10 ** ((elo_away + hfa - elo_home) / 400))


def wdl_probs(elo_home: float, elo_away: float, hfa: float,
              draw_base: float, draw_slope: float) -> tuple:
    e_home   = expected_score(elo_home, elo_away, hfa)
    elo_diff = abs(elo_home - elo_away)
    p_draw   = max(0.01, draw_base - draw_slope * elo_diff / 400)
    p_home   = max(0.01, e_home - p_draw / 2)
    p_away   = max(0.01, 1 - e_home - p_draw / 2)
    total    = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def normalise_pinnacle(home_odds: float, draw_odds: float, away_odds: float) -> tuple:
    raw        = np.array([1 / home_odds, 1 / draw_odds, 1 / away_odds])
    normalised = raw / raw.sum()
    return float(normalised[0]), float(normalised[1]), float(normalised[2])


def ig(p: float) -> float:
    return float(-np.log2(max(p, 1e-9)))


# ── Calibration ───────────────────────────────────────────────────────────────
def calibrate_draw_params(matches_df: pd.DataFrame,
                          elo_lookup: dict,
                          hfa: float = HFA_FIXED) -> tuple[float, float]:
    def log_loss(params):
        draw_base, draw_slope = params
        if draw_base <= 0 or draw_slope < 0 or draw_base > 1:
            return 1e9
        total_loss, n = 0.0, 0
        for _, row in matches_df.iterrows():
            prev_date = row["match_date"] - timedelta(days=1)
            eh = elo_lookup.get((row["home_team"], prev_date))
            ea = elo_lookup.get((row["away_team"], prev_date))
            if eh is None or ea is None:
                continue
            eh_elo, eh_hfa = eh
            ea_elo, _      = ea
            ph, pd_, pa = wdl_probs(eh_elo, ea_elo, eh_hfa, draw_base, draw_slope)
            p_actual = {"home_win": ph, "draw": pd_, "away_win": pa}.get(row["result"])
            if p_actual is None:
                continue
            total_loss -= np.log(max(p_actual, 1e-9))
            n += 1
        return total_loss / n if n > 0 else 1e9

    result = minimize(log_loss, x0=[0.26, 0.20], method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000})
    db, ds = result.x
    print(f"  draw_base={db:.4f}, draw_slope={ds:.4f} (log-loss={result.fun:.4f})")
    return float(db), float(ds)


# ── DB fetchers ───────────────────────────────────────────────────────────────
def fetch_elo_lookup(conn, table: str) -> dict:
    """Build {(team, date) -> (elo, hfa)} from elo1 or elo1_hfa."""
    has_hfa = (table == "elo1_hfa")
    if has_hfa:
        df = pd.read_sql_query(f"SELECT team, elo, date, hfa FROM {table}", conn)
    else:
        df = pd.read_sql_query(f"SELECT team, elo, date FROM {table}", conn)
        df["hfa"] = HFA_FIXED
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return {(row.team, row.date): (row.elo, row.hfa) for row in df.itertuples()}


def fetch_matches(conn) -> pd.DataFrame:
    query = """
        SELECT event_id, home_team, away_team,
               starts, home_score, away_score, result,
               starts::date AS match_date
        FROM results1
        WHERE league_id = %s
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=(LEAGUE_ID,))
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    df["result"]     = df["result"].astype(str).str.strip().str.lower()
    return df.drop_duplicates(subset=["event_id"])


def fetch_pinnacle_open_close(conn) -> pd.DataFrame:
    query = """
        WITH ranked AS (
            SELECT
                event_id, logged_time,
                home_odds, draw_odds, away_odds,
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
        print("Fetching data...")
        elo_fixed   = fetch_elo_lookup(conn, "elo1")
        elo_dynamic = fetch_elo_lookup(conn, "elo1_hfa")
        print(f"  Fixed-HFA Elo lookup  : {len(elo_fixed)} entries.")
        print(f"  Dynamic-HFA Elo lookup: {len(elo_dynamic)} entries.")

        matches  = fetch_matches(conn)
        pinnacle = fetch_pinnacle_open_close(conn)
        print(f"  Matches : {len(matches)}")
        print(f"  Pinnacle: {len(pinnacle)} events with odds.")

        # Calibrate draw params once using fixed-HFA Elos as the baseline.
        # The same params are reused for dynamic HFA so that HFA is the only
        # variable being compared between the two models.
        print("\nCalibrating draw parameters (shared, using fixed-HFA Elos)...")
        draw_base, draw_slope = calibrate_draw_params(matches, elo_fixed)
        draw_base_f = draw_base_d = draw_base
        draw_slope_f = draw_slope_d = draw_slope

        df = matches.merge(pinnacle, on="event_id", how="inner")
        print(f"\nBuilding analysis for {len(df)} matched events...")

        rows = []
        skipped = 0

        for _, row in df.iterrows():
            match_date = row["match_date"]
            prev_date  = match_date - timedelta(days=1)
            open_date  = row["open_date"]
            result     = row["result"]

            # ── Fixed HFA ────────────────────────────────────────────────────
            fh_o = elo_fixed.get((row["home_team"], open_date))
            fa_o = elo_fixed.get((row["away_team"], open_date))
            fh_c = elo_fixed.get((row["home_team"], prev_date))
            fa_c = elo_fixed.get((row["away_team"], prev_date))

            # ── Dynamic HFA ──────────────────────────────────────────────────
            dh_o = elo_dynamic.get((row["home_team"], open_date))
            da_o = elo_dynamic.get((row["away_team"], open_date))
            dh_c = elo_dynamic.get((row["home_team"], prev_date))
            da_c = elo_dynamic.get((row["away_team"], prev_date))

            if None in (fh_o, fa_o, fh_c, fa_c, dh_o, da_o, dh_c, da_c):
                skipped += 1
                continue

            # Unpack (elo, hfa) tuples
            elo_fh_o, hfa_fo = fh_o;  elo_fa_o, _      = fa_o
            elo_fh_c, hfa_fc = fh_c;  elo_fa_c, _      = fa_c
            elo_dh_o, hfa_do = dh_o;  elo_da_o, _      = da_o
            elo_dh_c, hfa_dc = dh_c;  elo_da_c, _      = da_c

            # Probabilities — fixed HFA
            fph_o, fpd_o, fpa_o = wdl_probs(elo_fh_o, elo_fa_o, hfa_fo, draw_base_f, draw_slope_f)
            fph_c, fpd_c, fpa_c = wdl_probs(elo_fh_c, elo_fa_c, hfa_fc, draw_base_f, draw_slope_f)

            # Probabilities — dynamic HFA
            dph_o, dpd_o, dpa_o = wdl_probs(elo_dh_o, elo_da_o, hfa_do, draw_base_d, draw_slope_d)
            dph_c, dpd_c, dpa_c = wdl_probs(elo_dh_c, elo_da_c, hfa_dc, draw_base_d, draw_slope_d)

            # Pinnacle normalised
            pin_ph_o, pin_pd_o, pin_pa_o = normalise_pinnacle(
                row["open_home_odds"], row["open_draw_odds"], row["open_away_odds"])
            pin_ph_c, pin_pd_c, pin_pa_c = normalise_pinnacle(
                row["close_home_odds"], row["close_draw_odds"], row["close_away_odds"])

            outcome_map = {"home_win": 0, "draw": 1, "away_win": 2}
            idx = outcome_map.get(result)
            if idx is None:
                skipped += 1
                continue

            fp_o  = (fph_o,   fpd_o,   fpa_o  )[idx]
            fp_c  = (fph_c,   fpd_c,   fpa_c  )[idx]
            dp_o  = (dph_o,   dpd_o,   dpa_o  )[idx]
            dp_c  = (dph_c,   dpd_c,   dpa_c  )[idx]
            pp_o  = (pin_ph_o, pin_pd_o, pin_pa_o)[idx]
            pp_c  = (pin_ph_c, pin_pd_c, pin_pa_c)[idx]

            rows.append((
                int(row["event_id"]),
                str(row["home_team"]), str(row["away_team"]),
                row["starts"],
                int(row["home_score"]), int(row["away_score"]),
                result,
                # Elo ratings (fixed then dynamic)
                int(elo_fh_o), int(elo_fa_o), int(elo_fh_c), int(elo_fa_c),
                int(elo_dh_o), int(elo_da_o), int(elo_dh_c), int(elo_da_c),
                round(hfa_do,3), round(hfa_dc,3),
                # Opening probs: fixed Elo, dynamic Elo, Pinnacle
                round(fph_o,4), round(fpd_o,4), round(fpa_o,4),
                round(dph_o,4), round(dpd_o,4), round(dpa_o,4),
                round(pin_ph_o,4), round(pin_pd_o,4), round(pin_pa_o,4),
                # Closing probs: fixed Elo, dynamic Elo, Pinnacle
                round(fph_c,4), round(fpd_c,4), round(fpa_c,4),
                round(dph_c,4), round(dpd_c,4), round(dpa_c,4),
                round(pin_ph_c,4), round(pin_pd_c,4), round(pin_pa_c,4),
                # Information gain: open (fixed, dynamic, pin), close (fixed, dynamic, pin)
                round(ig(fp_o),4), round(ig(dp_o),4), round(ig(pp_o),4),
                round(ig(fp_c),4), round(ig(dp_c),4), round(ig(pp_c),4),
                # Calibration params
                round(draw_base,4), round(draw_slope,4),
            ))

        print(f"  Computed: {len(rows)} | Skipped: {skipped}")

        # ── Ensure table has all columns ──────────────────────────────────────
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS match_odds_analysis (
                    event_id              BIGINT PRIMARY KEY,
                    home_team             VARCHAR(100),
                    away_team             VARCHAR(100),
                    starts                TIMESTAMP,
                    home_score            INTEGER,
                    away_score            INTEGER,
                    result                VARCHAR(100),

                    -- Elo ratings
                    elo_home_open         INTEGER,
                    elo_away_open         INTEGER,
                    elo_home_close        INTEGER,
                    elo_away_close        INTEGER,
                    dhfa_elo_home_open    INTEGER,
                    dhfa_elo_away_open    INTEGER,
                    dhfa_elo_home_close   INTEGER,
                    dhfa_elo_away_close   INTEGER,
                    dhfa_open             NUMERIC(7,3),
                    dhfa_close            NUMERIC(7,3),

                    -- Opening probabilities (all three models)
                    elo_open_p_home       NUMERIC(6,4),
                    elo_open_p_draw       NUMERIC(6,4),
                    elo_open_p_away       NUMERIC(6,4),
                    dhfa_open_p_home      NUMERIC(6,4),
                    dhfa_open_p_draw      NUMERIC(6,4),
                    dhfa_open_p_away      NUMERIC(6,4),
                    pin_open_p_home       NUMERIC(6,4),
                    pin_open_p_draw       NUMERIC(6,4),
                    pin_open_p_away       NUMERIC(6,4),

                    -- Closing probabilities (all three models)
                    elo_close_p_home      NUMERIC(6,4),
                    elo_close_p_draw      NUMERIC(6,4),
                    elo_close_p_away      NUMERIC(6,4),
                    dhfa_close_p_home     NUMERIC(6,4),
                    dhfa_close_p_draw     NUMERIC(6,4),
                    dhfa_close_p_away     NUMERIC(6,4),
                    pin_close_p_home      NUMERIC(6,4),
                    pin_close_p_draw      NUMERIC(6,4),
                    pin_close_p_away      NUMERIC(6,4),

                    -- Information gain (all three models, open and close)
                    ig_elo_open           NUMERIC(8,4),
                    ig_dhfa_open          NUMERIC(8,4),
                    ig_pin_open           NUMERIC(8,4),
                    ig_elo_close          NUMERIC(8,4),
                    ig_dhfa_close         NUMERIC(8,4),
                    ig_pin_close          NUMERIC(8,4),

                    -- Calibration params
                    draw_base             NUMERIC(6,4),
                    draw_slope            NUMERIC(6,4)
                )
            """)

        conn.commit()

        # ── Upsert ────────────────────────────────────────────────────────────
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO match_odds_analysis (
                    event_id, home_team, away_team, starts, home_score, away_score, result,
                    elo_home_open, elo_away_open, elo_home_close, elo_away_close,
                    dhfa_elo_home_open, dhfa_elo_away_open, dhfa_elo_home_close, dhfa_elo_away_close,
                    dhfa_open, dhfa_close,
                    elo_open_p_home, elo_open_p_draw, elo_open_p_away,
                    dhfa_open_p_home, dhfa_open_p_draw, dhfa_open_p_away,
                    pin_open_p_home, pin_open_p_draw, pin_open_p_away,
                    elo_close_p_home, elo_close_p_draw, elo_close_p_away,
                    dhfa_close_p_home, dhfa_close_p_draw, dhfa_close_p_away,
                    pin_close_p_home, pin_close_p_draw, pin_close_p_away,
                    ig_elo_open, ig_dhfa_open, ig_pin_open,
                    ig_elo_close, ig_dhfa_close, ig_pin_close,
                    draw_base, draw_slope
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,
                    %s,%s,%s, %s,%s,%s,
                    %s,%s,%s, %s,%s,%s,
                    %s,%s,%s, %s,%s,%s,
                    %s,%s,%s, %s,%s,%s,
                    %s,%s
                )
                ON CONFLICT (event_id) DO UPDATE SET
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
                    dhfa_elo_home_open  = EXCLUDED.dhfa_elo_home_open,
                    dhfa_elo_away_open  = EXCLUDED.dhfa_elo_away_open,
                    dhfa_elo_home_close = EXCLUDED.dhfa_elo_home_close,
                    dhfa_elo_away_close = EXCLUDED.dhfa_elo_away_close,
                    dhfa_open           = EXCLUDED.dhfa_open,
                    dhfa_close          = EXCLUDED.dhfa_close,
                    dhfa_open_p_home    = EXCLUDED.dhfa_open_p_home,
                    dhfa_open_p_draw    = EXCLUDED.dhfa_open_p_draw,
                    dhfa_open_p_away    = EXCLUDED.dhfa_open_p_away,
                    dhfa_close_p_home   = EXCLUDED.dhfa_close_p_home,
                    dhfa_close_p_draw   = EXCLUDED.dhfa_close_p_draw,
                    dhfa_close_p_away   = EXCLUDED.dhfa_close_p_away,
                    pin_open_p_home     = EXCLUDED.pin_open_p_home,
                    pin_open_p_draw     = EXCLUDED.pin_open_p_draw,
                    pin_open_p_away     = EXCLUDED.pin_open_p_away,
                    pin_close_p_home    = EXCLUDED.pin_close_p_home,
                    pin_close_p_draw    = EXCLUDED.pin_close_p_draw,
                    pin_close_p_away    = EXCLUDED.pin_close_p_away,
                    ig_elo_open         = EXCLUDED.ig_elo_open,
                    ig_elo_close        = EXCLUDED.ig_elo_close,
                    ig_dhfa_open        = EXCLUDED.ig_dhfa_open,
                    ig_dhfa_close       = EXCLUDED.ig_dhfa_close,
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

        # ── Summary ───────────────────────────────────────────────────────────
        summary = pd.read_sql_query("""
            SELECT
                ROUND(AVG(ig_elo_open),   4) AS elo_fixed_open,
                ROUND(AVG(ig_elo_close),  4) AS elo_fixed_close,
                ROUND(AVG(ig_dhfa_open),  4) AS elo_dynamic_open,
                ROUND(AVG(ig_dhfa_close), 4) AS elo_dynamic_close,
                ROUND(AVG(ig_pin_open),   4) AS pinnacle_open,
                ROUND(AVG(ig_pin_close),  4) AS pinnacle_close,
                COUNT(*) AS n_matches
            FROM match_odds_analysis
        """, conn)
        print("\n── Average Information Gain (lower = better) ──")
        print(summary.to_string(index=False))

    finally:
        conn.close()


if __name__ == "__main__":
    main()