import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

# Charger les données
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# Définition du modèle
model = RandomForestRegressor()

# Définition des paramètres
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20]
}

# GridSearch
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train.values.ravel())

# Sauvegarde des meilleurs paramètres
joblib.dump(grid_search.best_params_, "models/best_params.pkl")
