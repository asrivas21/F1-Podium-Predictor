import fastf1
import pandas as pd

# Enable Fast-F1's local cache
fastf1.Cache.enable_cache('f1_cache')

def load_race_data(year, gp_name, session_type='R'):
    """
    Loads the race session and lap data for a given Grand Prix.
    
    Returns:
    - laps: Pandas DataFrame of lap-by-lap data
    - session: FastF1 session object (used to access results/weather)
    """
    session = fastf1.get_session(year, gp_name, session_type)
    session.load()

    laps = session.laps
    laps = laps[laps["LapTime"].notna() & laps["IsAccurate"]]
    
    return laps, session

def load_multiple_races(year, gp_names, session_type='R'):
    """
    Loads laps and session metadata for multiple races.
    
    Returns:
    - combined_laps: DataFrame of all laps
    - all_sessions: list of (gp_name, session) tuples
    """
    all_laps = []
    all_sessions = []

    for gp in gp_names:
        try:
            print(f"⏳ Loading {gp} {year}...")
            session = fastf1.get_session(year, gp, session_type)
            session.load()
            laps = session.laps
            laps = laps[laps["LapTime"].notna() & laps["IsAccurate"]]
            laps["GrandPrix"] = gp
            all_laps.append(laps)
            all_sessions.append((gp, session))
            print(f"✅ Loaded {gp}")
        except Exception as e:
            print(f"❌ Failed to load {gp}: {e}")

    if not all_laps:
        raise ValueError("No races could be loaded.")

    combined_laps = pd.concat(all_laps, ignore_index=True)
    return combined_laps, all_sessions
