import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Charger les données
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# Charger les meilleurs paramètres
best_params = joblib.load("models/best_params.pkl")

# Entraînement du modèle
model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train.values.ravel())

# Sauvegarde du modèle
joblib.dump(model, "models/final_model.pkl")
