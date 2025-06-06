import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def prepare_multi_race_podium_data(all_laps, all_sessions):
    rows = []

    for gp_name, session in all_sessions:
        # Filter laps for this race
        race_laps = all_laps[all_laps["GrandPrix"] == gp_name]

        # Coerce numeric fields
        race_laps["Stint"] = pd.to_numeric(race_laps["Stint"], errors="coerce")
        race_laps["LapNumber"] = pd.to_numeric(race_laps["LapNumber"], errors="coerce")
        race_laps["TrackStatus"] = pd.to_numeric(race_laps["TrackStatus"], errors="coerce")
        race_laps["PitOutTime"] = pd.to_numeric(race_laps["PitOutTime"].notna(), errors="coerce")

        # Group by driver and extract per-driver race-level info
        driver_df = race_laps.groupby("Driver").agg({
            "Team": "first",
            "Stint": "max",
            "Compound": lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
            "LapNumber": "max",
            "PitOutTime": "sum",
            "TrackStatus": "mean"
        }).reset_index()

        # Add constant weather info
        weather = session.weather_data
        driver_df["AirTemp"] = weather["AirTemp"].mean()
        driver_df["TrackTemp"] = weather["TrackTemp"].mean()
        driver_df["Humidity"] = weather["Humidity"].mean()

        # Add GP name to track which race it came from
        driver_df["GrandPrix"] = gp_name

        # Add podium label
        results = session.results[["Abbreviation", "Position"]]
        results.rename(columns={"Abbreviation": "Driver"}, inplace=True)
        driver_df = driver_df.merge(results, on="Driver")
        driver_df["is_podium"] = driver_df["Position"].apply(lambda x: 1 if x <= 3 else 0)
        driver_df.drop(columns=["Position"], inplace=True)

        rows.append(driver_df)

    # Combine all races
    full_df = pd.concat(rows, ignore_index=True)

    # One-hot encode categorical columns
    cat = full_df[["Driver", "Team", "Compound"]].astype(str)
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded = encoder.fit_transform(cat)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat.columns))

    # Combine with numeric
    numeric = full_df.drop(columns=["Driver", "Team", "Compound", "is_podium", "GrandPrix"])
    X = pd.concat([encoded_df, numeric], axis=1)
    y = full_df["is_podium"]
    X.index = full_df["Driver"] + " - " + full_df["GrandPrix"]  # helpful for output

    return X, y, X.columns.tolist()

def prepare_podium_data(laps, sessions):
    """
    Prepare features and labels for podium prediction across multiple races.

    Args:
        laps: combined laps from multiple races
        sessions: list of (gp_name, session) tuples
    Returns:
        X: feature matrix
        y: binary labels (1 = podium finish, 0 = not)
        feature_names: list of feature column names
    """

    # Step 1: Get results from each session and label podium finishers
    all_results = []
    for gp_name, session in sessions:
        results = session.results
        if results is None or results.empty:
            continue
        results = results[["Abbreviation", "Position"]].copy()
        results["GrandPrix"] = gp_name
        results["Label"] = results["Position"].apply(lambda x: 1 if x in [1, 2, 3] else 0)
        all_results.append(results)

    podium_df = pd.concat(all_results, ignore_index=True)
    podium_df.rename(columns={"Abbreviation": "Driver"}, inplace=True)

    # Step 2: Merge lap data with podium labels
    df = laps.copy()
    df = df.merge(podium_df, on=["Driver", "GrandPrix"], how="left")
    df["Label"] = df["Label"].fillna(0)  # drivers who didn't finish = not podium

    # Step 3: Feature engineering
    df["Stint"] = pd.to_numeric(df["Stint"], errors="coerce")
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df["TrackStatus"] = pd.to_numeric(df["TrackStatus"], errors="coerce")
    df["PitOutTime"] = pd.to_numeric(df["PitOutTime"].notna(), errors="coerce")
    
    # Use only first lap of each stint per driver as snapshot
    snapshot_df = df.sort_values("LapNumber").groupby(["Driver", "GrandPrix"]).first().reset_index()

    # Select features
    feature_cols = [
        "Team", "Compound", "Stint", "LapNumber", "TrackStatus",
        "AirTemp", "TrackTemp", "Humidity", "PitOutTime"
    ]
    expected_cols = ["Driver", "GrandPrix", "Label"] + feature_cols
    existing_cols = [col for col in expected_cols if col in snapshot_df.columns]
    snapshot_df = snapshot_df[existing_cols].dropna()

    # One-hot encode categorical features
    encoded = pd.get_dummies(snapshot_df, columns=["Team", "Compound", "TrackStatus"])

    # Build final X, y
    y = encoded["Label"]
    X = encoded.drop(columns=["Driver", "GrandPrix", "Label"])
    feature_names = X.columns.tolist()

    # Set a useful index: "VER - Miami Grand Prix"
    X.index = snapshot_df["Driver"] + " - " + snapshot_df["GrandPrix"]

    return X, y, feature_names
