import soccerdata as sd

espn = sd.ESPN(leagues="ENG-Premier League", seasons="2025")
schedule = espn.read_schedule()

print(schedule.columns)
print(schedule.head())

# Note the list around schedule["game_id"].iloc[0]
match_id = schedule["game_id"].iloc[0]
matchsheet = espn.read_matchsheet(match_id=[match_id])

print(matchsheet.columns)
print(matchsheet.head())
