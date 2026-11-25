from model_pipeline import prepare_data

def test_prepare_data():
    # Vérifie que la fonction renvoie bien 4 éléments
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
