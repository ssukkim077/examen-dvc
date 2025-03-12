import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données
url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
df = pd.read_csv(url)

# Séparer les variables explicatives et la cible
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Split en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarde des datasets
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)