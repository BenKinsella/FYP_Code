import soccerdata as sd

# ESPN
print("ESPN leagues:")
print(sd.ESPN.available_leagues())

#FBref
print("FBref leagues:")
print(sd.FBref.available_leagues())

# FiveThirtyEight
print("\nFiveThirtyEight leagues:")
print(sd.FiveThirtyEight.available_leagues())

# FotMob
print("\nFotMob leagues:")
print(sd.FotMob.available_leagues())

# Sofascore
print("\nSofascore leagues:")
print(sd.Sofascore.available_leagues())

# MatchHistory (Football-Data.co.uk wrapper)
print("\nMatchHistory leagues:")
print(sd.MatchHistory.available_leagues())

