import argparse
import subprocess
from prefect import flow, task
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# ==========================================
# TÂCHES (TASKS)
# ==========================================


@task
def install_dependencies():
    print("Installation des dépendances...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)


@task
def format_code():
    print("Formatage du code avec black...")
    # On cible uniquement nos fichiers de projet selon le PDF
    subprocess.run(
        ["black", "model_pipeline.py", "main.py", "pipeline_prefect.py"], check=True
    )


@task
def check_quality():
    print("Vérification de la qualité avec flake8...")
    # --max-line-length=88 pour éviter que flake8 ne râle sur les lignes trop longues de black
    subprocess.run(
        [
            "flake8",
            "model_pipeline.py",
            "main.py",
            "pipeline_prefect.py",
            "--max-line-length=88",
            "--extend-ignore=E203",
        ],
        check=False,
    )


@task
def check_security():
    print("Vérification de la sécurité avec bandit...")
    subprocess.run(
        ["bandit", "-r", "model_pipeline.py", "main.py", "pipeline_prefect.py"],
        check=False,
    )


@task
def run_unit_tests():
    print("Exécution des tests unitaires...")
    subprocess.run(["pytest", "test_model.py"], check=True)


@task
def task_prepare_data(filepath):
    print("Préparation des données...")
    return prepare_data(filepath)


@task
def task_train_model(X_train, y_train):
    print("Entraînement du modèle...")
    return train_model(X_train, y_train)


@task
def task_save_model(model):
    print("Sauvegarde du modèle...")
    save_model(model)


@task
def task_load_model():
    print("Chargement du modèle...")
    return load_model()


@task
def task_evaluate_model(model, X_test, y_test):
    print("Évaluation du modèle...")
    return evaluate_model(model, X_test, y_test)


# ==========================================
# FLOWS
# ==========================================


@flow(name="code")
def code_flow():
    """Flow pour les vérifications de code"""
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="train")
def train_flow():
    """Flow pour préparer les données et entraîner"""
    X_train, X_test, y_train, y_test = task_prepare_data("Churn_Modelling.csv")
    model = task_train_model(X_train, y_train)
    task_save_model(model)


@flow(name="evaluate")
def evaluate_flow():
    """Flow pour charger et évaluer le modèle"""
    X_train, X_test, y_train, y_test = task_prepare_data("Churn_Modelling.csv")
    model = task_load_model()
    task_evaluate_model(model, X_test, y_test)


@flow(name="all")
def all_flow():
    """Flow global qui exécute tout"""
    # On lance les vérifications de code d'abord
    code_flow()
    # Puis le pipeline ML
    X_train, X_test, y_train, y_test = task_prepare_data("Churn_Modelling.csv")
    model = task_train_model(X_train, y_train)
    task_save_model(model)
    task_evaluate_model(model, X_test, y_test)


# ==========================================
# EXÉCUTION VIA CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument(
        "--flow",
        type=str,
        choices=["all", "train", "evaluate", "code"],
        required=True,
        help="Choix du flow à exécuter",
    )
    args = parser.parse_args()

    if args.flow == "all":
        all_flow()
    elif args.flow == "train":
        train_flow()
    elif args.flow == "evaluate":
        evaluate_flow()
    elif args.flow == "code":
        code_flow()
