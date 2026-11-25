"""
model_pipeline.py
Pipeline ML modulaire pour la prédiction du churn client.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Charge, nettoie, encode, splitte et normalise les données.
    """
    print(f"\n--- 1. Préparation des données ({filepath}) ---")
    df = pd.read_csv(filepath)

    # Suppression des colonnes inutiles
    cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Encodage des variables catégorielles (One-Hot Encoding)
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    # Séparation X et y
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    # Split (avec stratify pour garder les proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    features = X.columns.tolist()
    print(
        f"Données prêtes : {X_train_scaled.shape[0]} lignes d'entraînement, {X_test_scaled.shape[0]} lignes de test."
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features


def train_model(X_train, y_train):
    """
    Entraîne le modèle LinearSVC.
    """
    print("\n--- 2. Entraînement du modèle ---")
    model = LinearSVC(random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    print("Entraînement terminé !")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur le jeu de test.
    """
    print("\n--- 3. Évaluation du modèle ---")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print("\nRapport de classification :\n", classification_report(y_test, y_pred))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))


def save_model(model, scaler, features, path: str = "model.joblib"):
    """
    Sauvegarde le modèle, le scaler et les features avec joblib.
    """
    print(f"\n--- 4. Sauvegarde du modèle dans '{path}' ---")
    joblib.dump({"model": model, "scaler": scaler, "features": features}, path)
    print("Modèle sauvegardé avec succès !")


def load_model(path: str = "model.joblib"):
    """
    Charge un modèle sauvegardé.
    """
    print(f"\n--- Chargement du modèle depuis '{path}' ---")
    data = joblib.load(path)
    return data["model"], data["scaler"], data["features"]
