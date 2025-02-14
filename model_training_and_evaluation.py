import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

def load_data():
    """Load the preprocessed data."""
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')
    return X_train, X_val, X_test, y_train, y_val, y_test

def define_models():
    """Define the models to be used in the automated selection process."""
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    return models

def define_search_spaces():
    """Define the search spaces for each model."""
    search_spaces = {
        'RandomForest': {
            'n_estimators': Integer(10, 500),
            'max_depth': Integer(1, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 20)
        },
        'SVM': {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'kernel': Categorical(['rbf', 'poly', 'sigmoid'])
        },
        'XGBoost': {
            'n_estimators': Integer(10, 500),
            'max_depth': Integer(1, 20),
            'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
            'subsample': Real(0.1, 1.0),
            'colsample_bytree': Real(0.1, 1.0)
        }
    }
    return search_spaces

def bayesian_optimization(model, search_space, X_train, y_train, X_val, y_val):
    """Perform Bayesian optimization for hyperparameter tuning."""
    optimizer = BayesSearchCV(
        model,
        search_space,
        n_iter=50,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    optimizer.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = optimizer.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, optimizer.predict_proba(X_val)[:, 1])
    
    return optimizer.best_estimator_, accuracy, f1, auc

def train_and_evaluate_models(models, search_spaces, X_train, y_train, X_val, y_val):
    """Train and evaluate models using Bayesian optimization."""
    results = {}
    for model_name, model in models.items():
        print(f"Training and optimizing {model_name}...")
        best_model, accuracy, f1, auc = bayesian_optimization(
            model, search_spaces[model_name], X_train, y_train, X_val, y_val
        )
        results[model_name] = {
            'model': best_model,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_score': auc
        }
    return results

def select_best_model(results):
    """Select the best model based on AUC score."""
    best_model_name = max(results, key=lambda x: results[x]['auc_score'])
    return best_model_name, results[best_model_name]

def main():
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Define models and search spaces
    models = define_models()
    search_spaces = define_search_spaces()
    
    # Train and evaluate models
    results = train_and_evaluate_models(models, search_spaces, X_train, y_train, X_val, y_val)
    
    # Select the best model
    best_model_name, best_model_info = select_best_model(results)
    
    print("\nBest Model:")
    print(f"Model: {best_model_name}")
    print(f"Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"F1 Score: {best_model_info['f1_score']:.4f}")
    print(f"AUC Score: {best_model_info['auc_score']:.4f}")
    
    # Evaluate on test set
    best_model = best_model_info['model']
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    
    print("\nTest Set Performance:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"AUC Score: {test_auc:.4f}")
    
    # Save the best model
    import joblib
    joblib.dump(best_model, 'best_model.joblib')

if __name__ == "__main__":
    main()

