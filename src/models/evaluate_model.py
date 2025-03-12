import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json

# Charger les données
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Charger le modèle
model = joblib.load("models/final_model.pkl")

# Prédictions
y_pred = model.predict(X_test)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarde des métriques
metrics = {"MSE": mse, "R2": r2}
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f)

# Sauvegarde des prédictions
predictions = pd.DataFrame({"Actual": y_test.values.ravel(), "Predicted": y_pred})
predictions.to_csv("data/processed/predictions.csv", index=False)
