import psycopg2
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

elo_df = pd.read_csv("C:/Users/kinse/Downloads/2025-10-08.csv")

initial_elos = dict(zip(elo_df["Club"], elo_df["Elo"]))

class FootballElo:

     # ---------- DB helpers ----------
    def connect_db(self):
        if self.conn is None:
            self.conn = psycopg2.connect(self.database_url)

    def close_db(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # ---------- Data fetchers ----------
    def fetch_completed_odds_events(self):
        """Fetch finished matches from odds1x2; no scores here."""
        self.connect_db()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=3)
        LEAGUE_ID = 1980
        query = """
            SELECT event_id, home_team, away_team, league_id, league_name, starts, home_score, away_score, result
            FROM results1
            WHERE starts < %s
            AND league_id = %s
        """
        df = pd.read_sql_query(query, self.conn, params=(cutoff,LEAGUE_ID))
        # Normalised names for matching
        for col in ["home_team", "away_team", "league_name"]:
            df[col + "_lc"] = df[col].astype(str).str.strip().str.lower()
        df["starts_date"] = pd.to_datetime(df["starts"]).dt.date
        # Deduplicate by event_id to avoid multiple inserts
        df = df.sort_values("starts").drop_duplicates(subset=["event_id"])
        return df

    def __init__(self, init_elo=1500, hfa=42, k=20):
        self.elo = initial_elos.copy() if initial_elos is not None else {}  # team: current_elo
        self.history = []  # past updates
        self.init_elo = init_elo
        self.hfa = hfa
        self.k = k
        #self.database_url = database_url
    
    def expected(self, elo_home, elo_away):
        return 1 / (1 + 10**((elo_away + self.hfa - elo_home) / 400))
    
    def update(self, home_team, away_team, home_goals, away_goals, date, comp_weight=1.0):
        # Current ratings
        elo_home = self.elo.get(home_team, self.init_elo)
        elo_away = self.elo.get(away_team, self.init_elo)
        
        # Expected & actual
        E_home = self.expected(elo_home, elo_away)
        S_home = 1 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0
        
        # Goal multiplier (ClubElo style)
        goal_diff = abs(home_goals - away_goals)
        goal_mult = np.sqrt(goal_diff + 1)
        
        # Elo change
        delta_home = self.k * comp_weight * goal_mult * (S_home - E_home)
        delta_away = -delta_home
        
        # Update
        self.elo[home_team] = elo_home + delta_home
        self.elo[away_team] = elo_away + delta_away
        
        record = {
            'date': date, 'home': home_team, 'away': away_team,
            'home_g': home_goals, 'away_g': away_goals,
            'elo_home_pre': elo_home, 'elo_away_pre': elo_away,
            'delta_home': delta_home
        }
        self.history.append(record)
        
        return delta_home, self.elo[home_team], self.elo[away_team]

if __name__ == "__main__":
    DATABASE_URL = os.environ["DATABASE_URL"]
    # SEASON = "2025"
    # LEAGUE = "ENG-Premier League"  # or ENG-Premier League etc.

    updater = FootballElo(
        database_url=DATABASE_URL,
        # league=LEAGUE,
        # season=SEASON,
    )
    updater.update()
