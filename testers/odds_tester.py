"""
PL Odds Analysis Pipeline
--------------------------
1. Calibrates draw_base / draw_slope against historical PL results
2. For each PL match computes odds and IG for five models:
     - Base Elo          (opening + closing)
     - Dynamic-HFA Elo   (opening + closing)
     - Time-Decay Elo    (opening + closing)
     - Dixon-Coles Elo   (opening + closing)
     - Pinnacle          (opening + closing)
3. Upserts everything into match_odds_analysis (dataset 1) or match_odds_analysis2 (dataset 2)

Switch between datasets by changing DATASET at the top:
  DATASET = 1  →  results1 / elo1* / odds1x2  / match_odds_analysis
  DATASET = 2  →  results2 / elo2* / odds2     / match_odds_analysis2
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
# ── Dataset switch ────────────────────────────────────────────────────────────
DATASET = 1   # change to 2 to run on football-data.co.uk data

# ── Derived from DATASET ──────────────────────────────────────────────────────
RESULTS_TABLE  = "results1"             if DATASET == 1 else "results2"
ODDS_TABLE     = "odds1x2"              if DATASET == 1 else "odds2"
ELO_BASE       = "elo1"                 if DATASET == 1 else "elo2"
ELO_HFA        = "elo1_hfa"             if DATASET == 1 else "elo2_hfa"
ELO_DECAY      = "elo1_decay"           if DATASET == 1 else "elo2_decay"
ELO_DC         = "elo1_dc"              if DATASET == 1 else "elo2_dc"
ANALYSIS_TABLE = "match_odds_analysis"  if DATASET == 1 else "match_odds_analysis2"

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
                          elo_lookup: dict) -> tuple[float, float]:
    """Calibrate draw_base and draw_slope using fixed-HFA Elos as baseline."""
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
    """Build {(team, date) -> (elo, hfa)} from any elo table."""
    has_hfa = table in ("elo1_hfa", "elo2_hfa")
    if has_hfa:
        df = pd.read_sql_query(f"SELECT team, elo, date, hfa FROM {table}", conn)
    else:
        df = pd.read_sql_query(f"SELECT team, elo, date FROM {table}", conn)
        df["hfa"] = HFA_FIXED
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return {(row.team, row.date): (row.elo, row.hfa) for row in df.itertuples()}


def safe_fetch_elo_lookup(conn, table: str):
    """
    Same as fetch_elo_lookup but returns None if the table doesn't exist yet.
    Allows odds_analysis to run on pass 1 before DC/decay tables are populated,
    then pick them up automatically on pass 2.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (table,))
            exists = cur.fetchone()[0]
        if not exists:
            print(f"  {table} not found — skipping (run updater then rerun).")
            return None
        return fetch_elo_lookup(conn, table)
    except Exception as e:
        conn.rollback()
        print(f"  {table} could not be loaded ({e}) — skipping.")
        return None


def fetch_matches(conn) -> pd.DataFrame:
    query = f"""
        SELECT event_id, home_team, away_team,
               starts, home_score, away_score, result,
               starts::date AS match_date
        FROM {RESULTS_TABLE}
        WHERE league_id = %s
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=(LEAGUE_ID,))
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    df["result"]     = df["result"].astype(str).str.strip().str.lower()
    return df.drop_duplicates(subset=["event_id"])


def fetch_pinnacle_open_close(conn) -> pd.DataFrame:
    query = f"""
        WITH ranked AS (
            SELECT
                event_id, logged_time,
                home_odds, draw_odds, away_odds,
                ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY logged_time ASC)  AS rn_open,
                ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY logged_time DESC) AS rn_close
            FROM {ODDS_TABLE}
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


def get_probs(lookup: dict, home_team: str, away_team: str,
              lookup_date: date, draw_base: float, draw_slope: float) -> tuple:
    """Helper to get (p_home, p_draw, p_away) from any elo lookup for a given date."""
    eh = lookup.get((home_team, lookup_date))
    ea = lookup.get((away_team, lookup_date))
    if eh is None or ea is None:
        return None
    elo_h, hfa = eh
    elo_a, _   = ea
    return wdl_probs(elo_h, elo_a, hfa, draw_base, draw_slope)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        print(f"Running dataset {DATASET} → {RESULTS_TABLE} / {ODDS_TABLE} / {ANALYSIS_TABLE}")
        print("Fetching data...")
        elo_base  = fetch_elo_lookup(conn, ELO_BASE)
        elo_hfa   = fetch_elo_lookup(conn, ELO_HFA)
        elo_decay = safe_fetch_elo_lookup(conn, ELO_DECAY)
        elo_dc    = safe_fetch_elo_lookup(conn, ELO_DC)
        print(f"  Base Elo lookup        : {len(elo_base)} entries.")
        print(f"  Dynamic-HFA Elo lookup : {len(elo_hfa)} entries.")
        if elo_decay: print(f"  Time-Decay Elo lookup  : {len(elo_decay)} entries.")
        if elo_dc:    print(f"  Dixon-Coles Elo lookup : {len(elo_dc)} entries.")

        matches  = fetch_matches(conn)
        pinnacle = fetch_pinnacle_open_close(conn)
        print(f"  Matches : {len(matches)}")
        print(f"  Pinnacle: {len(pinnacle)} events with odds.")

        # Calibrate draw params once using base Elo — shared across all models
        print("\nCalibrating draw parameters (shared, using base Elo)...")
        draw_base, draw_slope = calibrate_draw_params(matches, elo_base)

        df = matches.merge(pinnacle, on="event_id", how="inner")
        print(f"\nBuilding analysis for {len(df)} matched events...")

        rows    = []
        skipped = 0

        for _, row in df.iterrows():
            match_date = row["match_date"]
            prev_date  = match_date - timedelta(days=1)
            open_date  = row["open_date"]
            result     = row["result"]

            # Get probs for each model at open and close
            # decay and dc may be None if their tables aren't populated yet
            base_o  = get_probs(elo_base,  row["home_team"], row["away_team"], open_date, draw_base, draw_slope)
            base_c  = get_probs(elo_base,  row["home_team"], row["away_team"], prev_date, draw_base, draw_slope)
            hfa_o   = get_probs(elo_hfa,   row["home_team"], row["away_team"], open_date, draw_base, draw_slope)
            hfa_c   = get_probs(elo_hfa,   row["home_team"], row["away_team"], prev_date, draw_base, draw_slope)
            decay_o = get_probs(elo_decay, row["home_team"], row["away_team"], open_date, draw_base, draw_slope) if elo_decay else None
            decay_c = get_probs(elo_decay, row["home_team"], row["away_team"], prev_date, draw_base, draw_slope) if elo_decay else None
            dc_o    = get_probs(elo_dc,    row["home_team"], row["away_team"], open_date, draw_base, draw_slope) if elo_dc    else None
            dc_c    = get_probs(elo_dc,    row["home_team"], row["away_team"], prev_date, draw_base, draw_slope) if elo_dc    else None

            if None in (base_o, base_c, hfa_o, hfa_c):
                skipped += 1
                continue

            # Pinnacle normalised probs
            pin_o = normalise_pinnacle(row["open_home_odds"],  row["open_draw_odds"],  row["open_away_odds"])
            pin_c = normalise_pinnacle(row["close_home_odds"], row["close_draw_odds"], row["close_away_odds"])

            # Index into probs tuple by outcome
            outcome_map = {"home_win": 0, "draw": 1, "away_win": 2}
            idx = outcome_map.get(result)
            if idx is None:
                skipped += 1
                continue

            def p3(probs):
                """Return rounded (h,d,a) tuple or (None,None,None) if probs is None."""
                if probs is None:
                    return None, None, None
                return round(probs[0],4), round(probs[1],4), round(probs[2],4)

            def ig_or_none(probs, i):
                return round(ig(probs[i]),4) if probs is not None else None

            rows.append((
                int(row["event_id"]),
                str(row["home_team"]), str(row["away_team"]),
                row["starts"],
                int(row["home_score"]), int(row["away_score"]),
                result,

                # ── Opening probabilities: all five models ──
                *p3(base_o),
                *p3(hfa_o),
                *p3(decay_o),
                *p3(dc_o),
                *p3(pin_o),

                # ── Closing probabilities: all five models ──
                *p3(base_c),
                *p3(hfa_c),
                *p3(decay_c),
                *p3(dc_c),
                *p3(pin_c),

                # ── Information gain: open then close, all five models ──
                ig_or_none(base_o,  idx), ig_or_none(hfa_o,   idx),
                ig_or_none(decay_o, idx), ig_or_none(dc_o,    idx),
                ig_or_none(pin_o,   idx),
                ig_or_none(base_c,  idx), ig_or_none(hfa_c,   idx),
                ig_or_none(decay_c, idx), ig_or_none(dc_c,    idx),
                ig_or_none(pin_c,   idx),

                # ── Calibration params ──
                round(draw_base,4), round(draw_slope,4),
            ))

        print(f"  Computed: {len(rows)} | Skipped: {skipped}")

        # ── Create / update table ─────────────────────────────────────────────
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {ANALYSIS_TABLE} (
                    event_id              BIGINT PRIMARY KEY,
                    home_team             VARCHAR(100),
                    away_team             VARCHAR(100),
                    starts                TIMESTAMP,
                    home_score            INTEGER,
                    away_score            INTEGER,
                    result                VARCHAR(100),

                    -- Opening probabilities (home / draw / away) per model
                    base_open_p_home      NUMERIC(6,4),
                    base_open_p_draw      NUMERIC(6,4),
                    base_open_p_away      NUMERIC(6,4),
                    hfa_open_p_home       NUMERIC(6,4),
                    hfa_open_p_draw       NUMERIC(6,4),
                    hfa_open_p_away       NUMERIC(6,4),
                    decay_open_p_home     NUMERIC(6,4),
                    decay_open_p_draw     NUMERIC(6,4),
                    decay_open_p_away     NUMERIC(6,4),
                    dc_open_p_home        NUMERIC(6,4),
                    dc_open_p_draw        NUMERIC(6,4),
                    dc_open_p_away        NUMERIC(6,4),
                    pin_open_p_home       NUMERIC(6,4),
                    pin_open_p_draw       NUMERIC(6,4),
                    pin_open_p_away       NUMERIC(6,4),

                    -- Closing probabilities (home / draw / away) per model
                    base_close_p_home     NUMERIC(6,4),
                    base_close_p_draw     NUMERIC(6,4),
                    base_close_p_away     NUMERIC(6,4),
                    hfa_close_p_home      NUMERIC(6,4),
                    hfa_close_p_draw      NUMERIC(6,4),
                    hfa_close_p_away      NUMERIC(6,4),
                    decay_close_p_home    NUMERIC(6,4),
                    decay_close_p_draw    NUMERIC(6,4),
                    decay_close_p_away    NUMERIC(6,4),
                    dc_close_p_home       NUMERIC(6,4),
                    dc_close_p_draw       NUMERIC(6,4),
                    dc_close_p_away       NUMERIC(6,4),
                    pin_close_p_home      NUMERIC(6,4),
                    pin_close_p_draw      NUMERIC(6,4),
                    pin_close_p_away      NUMERIC(6,4),

                    -- Information gain (opening)
                    ig_base_open          NUMERIC(8,4),
                    ig_hfa_open           NUMERIC(8,4),
                    ig_decay_open         NUMERIC(8,4),
                    ig_dc_open            NUMERIC(8,4),
                    ig_pin_open           NUMERIC(8,4),

                    -- Information gain (closing)
                    ig_base_close         NUMERIC(8,4),
                    ig_hfa_close          NUMERIC(8,4),
                    ig_decay_close        NUMERIC(8,4),
                    ig_dc_close           NUMERIC(8,4),
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
                f"""
                INSERT INTO {ANALYSIS_TABLE} (
                    event_id, home_team, away_team, starts, home_score, away_score, result,
                    base_open_p_home,  base_open_p_draw,  base_open_p_away,
                    hfa_open_p_home,   hfa_open_p_draw,   hfa_open_p_away,
                    decay_open_p_home, decay_open_p_draw, decay_open_p_away,
                    dc_open_p_home,    dc_open_p_draw,    dc_open_p_away,
                    pin_open_p_home,   pin_open_p_draw,   pin_open_p_away,
                    base_close_p_home,  base_close_p_draw,  base_close_p_away,
                    hfa_close_p_home,   hfa_close_p_draw,   hfa_close_p_away,
                    decay_close_p_home, decay_close_p_draw, decay_close_p_away,
                    dc_close_p_home,    dc_close_p_draw,    dc_close_p_away,
                    pin_close_p_home,   pin_close_p_draw,   pin_close_p_away,
                    ig_base_open,  ig_hfa_open,   ig_decay_open,  ig_dc_open,  ig_pin_open,
                    ig_base_close, ig_hfa_close,  ig_decay_close, ig_dc_close, ig_pin_close,
                    draw_base, draw_slope
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s, %s,%s,%s, %s,%s,%s, %s,%s,%s, %s,%s,%s,
                    %s,%s,%s, %s,%s,%s, %s,%s,%s, %s,%s,%s, %s,%s,%s,
                    %s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,
                    %s,%s
                )
                ON CONFLICT (event_id) DO UPDATE SET
                    result             = EXCLUDED.result,
                    base_open_p_home   = EXCLUDED.base_open_p_home,
                    base_open_p_draw   = EXCLUDED.base_open_p_draw,
                    base_open_p_away   = EXCLUDED.base_open_p_away,
                    hfa_open_p_home    = EXCLUDED.hfa_open_p_home,
                    hfa_open_p_draw    = EXCLUDED.hfa_open_p_draw,
                    hfa_open_p_away    = EXCLUDED.hfa_open_p_away,
                    decay_open_p_home  = EXCLUDED.decay_open_p_home,
                    decay_open_p_draw  = EXCLUDED.decay_open_p_draw,
                    decay_open_p_away  = EXCLUDED.decay_open_p_away,
                    dc_open_p_home     = EXCLUDED.dc_open_p_home,
                    dc_open_p_draw     = EXCLUDED.dc_open_p_draw,
                    dc_open_p_away     = EXCLUDED.dc_open_p_away,
                    pin_open_p_home    = EXCLUDED.pin_open_p_home,
                    pin_open_p_draw    = EXCLUDED.pin_open_p_draw,
                    pin_open_p_away    = EXCLUDED.pin_open_p_away,
                    base_close_p_home  = EXCLUDED.base_close_p_home,
                    base_close_p_draw  = EXCLUDED.base_close_p_draw,
                    base_close_p_away  = EXCLUDED.base_close_p_away,
                    hfa_close_p_home   = EXCLUDED.hfa_close_p_home,
                    hfa_close_p_draw   = EXCLUDED.hfa_close_p_draw,
                    hfa_close_p_away   = EXCLUDED.hfa_close_p_away,
                    decay_close_p_home = EXCLUDED.decay_close_p_home,
                    decay_close_p_draw = EXCLUDED.decay_close_p_draw,
                    decay_close_p_away = EXCLUDED.decay_close_p_away,
                    dc_close_p_home    = EXCLUDED.dc_close_p_home,
                    dc_close_p_draw    = EXCLUDED.dc_close_p_draw,
                    dc_close_p_away    = EXCLUDED.dc_close_p_away,
                    pin_close_p_home   = EXCLUDED.pin_close_p_home,
                    pin_close_p_draw   = EXCLUDED.pin_close_p_draw,
                    pin_close_p_away   = EXCLUDED.pin_close_p_away,
                    ig_base_open       = EXCLUDED.ig_base_open,
                    ig_hfa_open        = EXCLUDED.ig_hfa_open,
                    ig_decay_open      = EXCLUDED.ig_decay_open,
                    ig_dc_open         = EXCLUDED.ig_dc_open,
                    ig_pin_open        = EXCLUDED.ig_pin_open,
                    ig_base_close      = EXCLUDED.ig_base_close,
                    ig_hfa_close       = EXCLUDED.ig_hfa_close,
                    ig_decay_close     = EXCLUDED.ig_decay_close,
                    ig_dc_close        = EXCLUDED.ig_dc_close,
                    ig_pin_close       = EXCLUDED.ig_pin_close,
                    draw_base          = EXCLUDED.draw_base,
                    draw_slope         = EXCLUDED.draw_slope
                """,
                rows,
                page_size=200,
            )
        conn.commit()
        print(f"Done! {len(rows)} rows upserted into {ANALYSIS_TABLE}.")

        # ── Summary ───────────────────────────────────────────────────────────
        summary = pd.read_sql_query(f"""
            SELECT
                ROUND(AVG(ig_base_open),   4) AS base_open,
                ROUND(AVG(ig_hfa_open),    4) AS hfa_open,
                ROUND(AVG(ig_decay_open),  4) AS decay_open,
                ROUND(AVG(ig_dc_open),     4) AS dc_open,
                ROUND(AVG(ig_pin_open),    4) AS pin_open,
                ROUND(AVG(ig_base_close),  4) AS base_close,
                ROUND(AVG(ig_hfa_close),   4) AS hfa_close,
                ROUND(AVG(ig_decay_close), 4) AS decay_close,
                ROUND(AVG(ig_dc_close),    4) AS dc_close,
                ROUND(AVG(ig_pin_close),   4) AS pin_close,
                COUNT(*) AS n_matches
            FROM {ANALYSIS_TABLE}
        """, conn)
        print("\n── Average Information Gain (lower = better) ──")
        print(summary.to_string(index=False))

        # ── First half vs second half split (by match date) ───────────────────
        half_split = pd.read_sql_query(f"""
            WITH ordered AS (
                SELECT
                    ig_base_open, ig_pin_open,
                    ig_base_close, ig_pin_close,
                    ROW_NUMBER() OVER (ORDER BY starts) AS rn,
                    COUNT(*) OVER () AS total
                FROM {ANALYSIS_TABLE}
            )
            SELECT
                CASE WHEN rn <= total / 2 THEN 'first_half' ELSE 'second_half' END AS half,
                ROUND(AVG(ig_base_open),  4) AS base_open,
                ROUND(AVG(ig_pin_open),   4) AS pin_open,
                ROUND(AVG(ig_base_close), 4) AS base_close,
                ROUND(AVG(ig_pin_close),  4) AS pin_close,
                COUNT(*) AS n_matches
            FROM ordered
            GROUP BY half
            ORDER BY half
        """, conn)
        print("\n── Base IG: First half vs Second half (lower = better) ──")
        print(half_split.to_string(index=False))

        # ── Rebuild base table (base Elo + Pinnacle only) ─────────────────────
        # Used by DC and decay updaters to fetch shared draw_base / draw_slope.
        base_table = f"{ANALYSIS_TABLE}_base"
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {base_table}")
            cur.execute(f"""
                CREATE TABLE {base_table} AS
                SELECT
                    event_id, home_team, away_team, starts,
                    home_score, away_score, result,
                    base_open_p_home,  base_open_p_draw,  base_open_p_away,
                    pin_open_p_home,   pin_open_p_draw,   pin_open_p_away,
                    base_close_p_home, base_close_p_draw, base_close_p_away,
                    pin_close_p_home,  pin_close_p_draw,  pin_close_p_away,
                    ig_base_open,  ig_pin_open,
                    ig_base_close, ig_pin_close,
                    draw_base, draw_slope
                FROM {ANALYSIS_TABLE}
            """)
            cur.execute(f"ALTER TABLE {base_table} ADD PRIMARY KEY (event_id)")
        conn.commit()
        print(f"Rebuilt {base_table} ({len(rows)} rows).")

    finally:
        conn.close()


if __name__ == "__main__":
    main()