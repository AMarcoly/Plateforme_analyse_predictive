import pandas as pd
from data_preprocessing import DataPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils import print_model_scores, plot_model_comparison, print_performance_analysis
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelTraining:
    def __init__(self, data_train, data_test):
        self.data_train = data_train
        self.data_test = data_test
        self.scores = {}

    def _align_features(self, X_train, X_test):
        """Aligne les colonnes entre train et test"""
        # Garde seulement les colonnes communes
        common_cols = X_train.columns.intersection(X_test.columns)
        return X_train[common_cols], X_test[common_cols]

    def _train_and_evaluate(self, model, model_name):
        """Méthode unique pour l'entraînement et l'évaluation"""
        # Préparation des données
        X_train = self.data_train.drop('Survived', axis=1)
        y_train = self.data_train['Survived']
        X_test = self.data_test.drop('Survived', axis=1)
        y_test = self.data_test['Survived']
        
        # Alignement des features (SUPPRIMER _prepare_features)
        X_train, X_test = self._align_features(X_train, X_test)
        
        # Entraînement
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Évaluation
        scores = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0)
        }
        
        self.scores[model_name] = scores
        return model, y_pred

    def logistic_model(self):
        model = LogisticRegression(max_iter=1000, random_state=42)
        return self._train_and_evaluate(model, 'LogisticRegression')

    def random_forest_model(self):
        model = RandomForestClassifier(random_state=42)
        return self._train_and_evaluate(model, 'RandomForest')

    def SVM_model(self):
        model = SVC(probability=True, random_state=42)
        return self._train_and_evaluate(model, 'SVM')

    def get_scores(self):
        return self.scores

# Usage inchangé dans le main
if __name__ == "__main__":
    # Chargement et nettoyage
    processor_train = DataPreprocessor("data_titanic/train.csv")
    processor_test = DataPreprocessor("data_titanic/test.csv")
    processor_res = pd.read_csv("data_titanic/gender_submission.csv")  

    clean_train = processor_train.clean_data()
    clean_test = processor_test.clean_data()

    # Récupérer les IDs des passagers de test
    test_ids = processor_test.get_passenger_ids()

    # Fusion des résultats
    clean_test_with_labels = pd.merge(
        test_ids.to_frame(),  # Convertir Series en DataFrame
        processor_res[['PassengerId', 'Survived']],
        on='PassengerId'
    )

    clean_test_with_labels = pd.concat([
        clean_test, 
        clean_test_with_labels['Survived']
    ], axis=1)


    # Modèle
    trainer = ModelTraining(clean_train, clean_test_with_labels)
    trainer.logistic_model()
    trainer.random_forest_model()
    trainer.SVM_model()

    # Résultats
    scores = trainer.get_scores()
    # Affichage
    df_scores = print_model_scores(scores)
    print_performance_analysis(scores)
    plot_model_comparison(scores)
    df_scores.to_csv('model_performance.csv')