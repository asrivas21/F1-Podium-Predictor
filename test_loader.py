from data_loader import load_race_data

if __name__ == "__main__":
    year = 2023
    gp = "Spanish Grand Prix"
    
    laps, session = load_race_data(year, gp)

    print("✅ Loaded session:", session.event['EventName'])
    print("Number of valid laps:", len(laps))
    print("Drivers in dataset:", laps['Driver'].unique())
    print(laps.head())
