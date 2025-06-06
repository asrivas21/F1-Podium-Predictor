from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

def train_podium_classifier(X, y, feature_names, test_size=0.3, random_state=42, return_probs=False):
    """
    Train a Random Forest classifier to predict whether a driver finishes on the podium.

    Args:
        X: feature matrix with driver identifiers in index (e.g. "VER - Miami GP")
        y: binary target (1 = podium, 0 = not)
        feature_names: list of column names
        return_probs: if True, also return prediction DataFrame with probabilities
    Returns:
        trained model (and optionally podium_probs DataFrame)
    """

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Evaluation table
    podium_probs = pd.DataFrame({
        "Driver": X_test.index,
        "Podium_Prob": y_prob,
        "Predicted_Label": y_pred,
        "True_Label": y_test.values
    })

    # Show predictions
    print("\n🏁 Predicted Podium Ranking (Top 5):")
    print(podium_probs.sort_values("Podium_Prob", ascending=False).head(5))

    # Metrics
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Feature importances
    print("\n🔥 Feature Importances:")
    importances = clf.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    for idx in sorted_idx[:15]:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")

   
    

    # 🏆 Analyze predicted podiums
    top_podiums = podium_probs[podium_probs["Predicted_Label"] == 1].copy()
    top_podiums["Actual_Driver"] = top_podiums["Driver"].apply(lambda x: x.split(" - ")[0])
    top_podiums["GrandPrix"] = top_podiums["Driver"].apply(lambda x: x.split(" - ")[1])

    print("\n🏁 Top Podium Probabilities (all races combined):")
    print(top_podiums[["Actual_Driver", "GrandPrix", "Podium_Prob", "Predicted_Label", "True_Label"]].head(10))

    # Count predicted podiums per driver
    # 🏆 Correct predicted podium count per driver
    # Extract just driver codes (e.g., 'VER' from 'VER - Hungarian Grand Prix')
    top_podiums["DriverName"] = top_podiums["Actual_Driver"].str.extract(r"^(\w+)", expand=False)

    # Count how many times each driver appeared on the predicted podium
    predicted_totals = top_podiums["DriverName"].value_counts().sort_values(ascending=False)

    print("\n🏆 Predicted Podium Totals Per Driver:")
    print(predicted_totals)

    # 🏆 Actual podium totals from test set
    true_podiums = podium_probs[podium_probs["True_Label"] == 1].copy()
    true_podiums["Actual_Driver"] = true_podiums["Driver"].apply(lambda x: x.split(" - ")[0])
    actual_totals = true_podiums["Actual_Driver"].value_counts().sort_values(ascending=False)

    print("\n🎯 Actual Podium Totals Per Driver:")
    print(actual_totals)


 # Return as requested
    if return_probs:
        return clf, podium_probs
    else:
        return clf
