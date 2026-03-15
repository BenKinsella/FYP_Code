# Results_data1

## Overview
This project is designed to analyze Premier League (PL) match data, compute probabilities and information gain (IG) for various models, and store the results in a PostgreSQL database. The pipeline includes fetching match data, calibrating parameters, and comparing the performance of different models.

## Features
- **Elo Ratings**: Computes probabilities using fixed and dynamic Home Field Advantage (HFA) Elo models.
- **Pinnacle Odds**: Normalizes Pinnacle odds and compares them with Elo-based probabilities.
- **Information Gain (IG)**: Measures the calibration quality of each model.
- **Database Integration**: Results are upserted into PostgreSQL tables for further analysis.
- **Dynamic HFA Updates**: Implements ClubElo's method for dynamically adjusting HFA based on match outcomes.

## Project Structure

## Key Scripts
- **`summary_tables/odds_summary1.py`**: The main pipeline for analyzing match data and computing probabilities for fixed-HFA, dynamic-HFA, and Pinnacle models.
- **`elo_updaters/update_elo1.py`**: Updates Elo ratings with a fixed HFA value.
- **`elo_updaters/update_eloHFA.py`**: Updates Elo ratings with dynamically adjusted HFA.
- **`oddsportal_scraper/odds_scraper.py`**: Scrapes match data from OddsPortal and updates the database.

## Database Tables
- **`results1`**: Stores match results, including scores and outcomes.
- **`elo1`**: Stores fixed-HFA Elo ratings for teams.
- **`elo1_hfa`**: Stores dynamic-HFA Elo ratings and HFA values.
- **`match_odds_analysis1`**: Stores computed probabilities and IG for all models.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/BenKinsella/Results_data1.git
   cd Results_data1
