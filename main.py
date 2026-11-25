"""
main.py
Point d'entrée du pipeline ML via arguments CLI.
"""

import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(
        description="Pipeline ML pour la prédiction du Churn"
    )
    parser.add_argument("--data", required=True, help="Chemin vers le fichier CSV")
    parser.add_argument(
        "--model-path",
        default="model.joblib",
        help="Chemin du fichier de sauvegarde du modèle",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate"],
        default="train",
        help="Action à réaliser",
    )

    args = parser.parse_args()

    if args.mode == "train":
        print("=== DÉMARRAGE DU PIPELINE D'ENTRAÎNEMENT ===")
        # Appel des fonctions dans l'ordre
        X_train, X_test, y_train, y_test, scaler, features = prepare_data(args.data)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model, scaler, features, args.model_path)
        print("\n=== PIPELINE TERMINÉ AVEC SUCCÈS ===")

    elif args.mode == "evaluate":
        print("=== DÉMARRAGE DE L'ÉVALUATION ===")
        # Chargement du modèle existant et évaluation
        model, scaler, features = load_model(args.model_path)
        _, X_test, _, y_test, _, _ = prepare_data(args.data)
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
