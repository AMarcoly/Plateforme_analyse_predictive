import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class HyperparameterTuning:
    def __init__(self, X_train, y_train, cv_folds=5, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_models = {}
        self.tuning_results = {}

    def tune_logistic_regression(self):
        param_grid = {
            'C': [0.1, 0.5, 1, 2, 5],  # Valeurs plus serrées autour de 1
            'penalty': ['l2'],  # Tester seulement L2
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [2000]
        }
        
        model = LogisticRegression(random_state=self.random_state)
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv_folds, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_models['LogisticRegression'] = grid_search.best_estimator_
        self.tuning_results['LogisticRegression'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        return grid_search.best_estimator_

    def tune_random_forest(self):
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        
        model = RandomForestClassifier(random_state=self.random_state)
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=50, cv=self.cv_folds,
            scoring='accuracy', n_jobs=-1, random_state=self.random_state, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        self.best_models['RandomForest'] = random_search.best_estimator_
        self.tuning_results['RandomForest'] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_
        }
        
        return random_search.best_estimator_

    def tune_all_models(self):
        print("Starting hyperparameter tuning for all models...")
        
        print("\nTuning Logistic Regression...")
        self.tune_logistic_regression()
        
        print("\nTuning Random Forest...")
        self.tune_random_forest()
        
        return self.best_models

    def evaluate_tuned_models(self, X_test, y_test):
        evaluation_results = {}
        
        for model_name, model in self.best_models.items():
            y_pred = model.predict(X_test)
            
            evaluation_results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
        
        return evaluation_results

    def get_best_params_summary(self):
        summary = {}
        for model_name, results in self.tuning_results.items():
            summary[model_name] = {
                'best_parameters': results['best_params'],
                'cross_val_score': results['best_score']
            }
        return summary

    def save_best_models(self, filepath='best_models/'):
        # Créer le dossier s'il n'existe pas
        os.makedirs(filepath, exist_ok=True)
        
        for model_name, model in self.best_models.items():
            filename = f"{filepath}{model_name}_best_model.pkl"
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")

def integrate_tuning_in_main():
    """
    Integration avec la structure main.py existante
    """
    from data_preprocessing import DataPreprocessor
    from model_training import ModelTraining
    
    # Chargement et prétraitement
    processor_train = DataPreprocessor("data_titanic/train.csv")
    processor_test = DataPreprocessor("data_titanic/test.csv") 
    processor_res = pd.read_csv("data_titanic/gender_submission.csv")
    
    clean_train = processor_train.clean_data()
    clean_test = processor_test.clean_data()
    
    # Préparation des données de test avec labels
    test_ids = processor_test.get_passenger_ids()
    clean_test_with_labels = pd.merge(
        test_ids.to_frame(),
        processor_res[['PassengerId', 'Survived']], 
        on='PassengerId'
    )
    clean_test_with_labels = pd.concat([
        clean_test, 
        clean_test_with_labels['Survived']
    ], axis=1)
    
    # Préparation des features pour le tuning
    X_train = clean_train.drop('Survived', axis=1)
    y_train = clean_train['Survived']
    X_test = clean_test_with_labels.drop('Survived', axis=1) 
    y_test = clean_test_with_labels['Survived']
    
    # Alignement des features
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Tuning des hyperparamètres
    tuner = HyperparameterTuning(X_train, y_train)
    best_models = tuner.tune_all_models()
    
    # Évaluation des modèles tunés
    tuned_results = tuner.evaluate_tuned_models(X_test, y_test)
    
    # Comparaison avec baseline
    baseline_trainer = ModelTraining(clean_train, clean_test_with_labels)
    baseline_trainer.logistic_model()
    baseline_trainer.random_forest_model()
    baseline_results = baseline_trainer.get_scores()
    
    # Affichage comparaison
    from utils import print_model_scores
    print("\nBASELINE PERFORMANCE:")
    print_model_scores(baseline_results)
    
    print("\nTUNED PERFORMANCE:")
    print_model_scores(tuned_results)
    
    # Résumé des meilleurs paramètres
    print("\nBEST PARAMETERS:")
    for model_name, params in tuner.get_best_params_summary().items():
        print(f"\n{model_name}:")
        print(f"  Best score: {params['cross_val_score']:.4f}")
        print(f"  Parameters: {params['best_parameters']}")
    
    # Sauvegarde des modèles
    tuner.save_best_models()
    
    return tuner, baseline_results, tuned_results

if __name__ == "__main__":
    integrate_tuning_in_main()