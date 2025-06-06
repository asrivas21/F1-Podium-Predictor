import fastf1
from fastf1 import plotting
import pandas as pd

fastf1.Cache.enable_cache('f1_cache')

def load_sessions(season, races):
    sessions = []
    for race in races:
        try:
            session = fastf1.get_session(season, race, 'R')
            session.load()
            sessions.append(session)
        except Exception as e:
            print(f"Skipping {race} due to error: {e}")
    return sessions
