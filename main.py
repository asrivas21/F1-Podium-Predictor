from data_loader import load_multiple_races
from prepare_data import prepare_podium_data
from models.predict_podium import train_podium_classifier
import pandas as pd

# === SETTINGS ===
year = 2023
gp_names = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
    "Azerbaijan Grand Prix", "Miami Grand Prix", "Spanish Grand Prix",
    "Canadian Grand Prix", "British Grand Prix", "Hungarian Grand Prix",
    "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix",
    "Singapore Grand Prix", "Japanese Grand Prix", "Qatar Grand Prix"
]

# === STEP 1: Load Data ===
print("🚦 Loading race data...")
laps, sessions = load_multiple_races(year, gp_names)
print(f"✅ Loaded data for {len(sessions)} races and {len(laps)} laps total")

# === STEP 2: Prepare Features ===
print("🧹 Preparing dataset...")
X, y, feature_names = prepare_podium_data(laps, sessions)
print(f"✅ Prepared dataset with {X.shape[0]} samples and {X.shape[1]} features")

# === STEP 3: Train and Predict ===
print("🤖 Training model and predicting podiums...")
model, podium_probs = train_podium_classifier(X, y, feature_names, return_probs=True)

# === STEP 4: Show top drivers from last race (or use aggregated data) ===
print("\n🏁 Top Podium Probabilities (all races combined):")
top_preds = podium_probs.sort_values("Podium_Prob", ascending=False).head(10)
top_preds["Actual_Driver"] = top_preds["Driver"].str.extract(r'^(\w+)')
top_preds["GrandPrix"] = top_preds["Driver"].str.extract(r'- (.+)$')
print(top_preds[["Actual_Driver", "GrandPrix", "Podium_Prob", "Predicted_Label", "True_Label"]])

# === OPTIONAL STEP: Aggregate by driver for season summary ===
print("\n📈 Estimated Podium Counts:")
summary = podium_probs.groupby("Driver")["Predicted_Label"].sum().sort_values(ascending=False)
print(summary.to_string())



# How many podiums the model thinks each driver should have
print("\n🏆 Predicted Podium Totals Per Driver:")
podium_counts = top_preds[top_preds["Predicted_Label"] == 1]["Driver"].value_counts()
print(podium_counts)

