import pandas as pd
from data_preprocessing import DataPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelTraining:
    def __init__(self, data_train, data_test):
        self.data_train = data_train
        self.data_test = data_test
        self.scores = {}

    def _prepare_features(self, X):
        # Encodage one-hot
        cols_to_encode = ['Embarked', 'Title', 'Deck']
        X = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)
        return X

    def fit_predict_evaluate(self, model, X_train, y_train, X_test, y_test=None):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if y_test is not None:
            scores = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            }
            return y_pred, scores
        return y_pred, None

    def logistic_model(self):
        X_train = self.data_train.drop('Survived', axis=1)
        y_train = self.data_train['Survived']
        X_test = self.data_test.drop('Survived', axis=1)
        y_test = self.data_test['Survived']
        X_train = self._prepare_features(X_train)
        X_test = self._prepare_features(X_test)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        model = LogisticRegression(max_iter=2500, solver='lbfgs')
        y_pred, scores = self._perform_and_evaluate(model, X_train, y_train, X_test, y_test)
        if scores:
            self.scores['LogisticRegression'] = scores
        return model, y_pred

    def random_forest_model(self):
        X_train = self.data_train.drop('Survived', axis=1)
        y_train = self.data_train['Survived']
        X_test = self.data_test.drop('Survived', axis=1)
        y_test = self.data_test['Survived']
        X_train = self._prepare_features(X_train)
        X_test = self._prepare_features(X_test)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        model = RandomForestClassifier()
        y_pred, scores = self._perform_and_evaluate(model, X_train, y_train, X_test, y_test)
        if scores:
            self.scores['RandomForest'] = scores
        return model, y_pred

    def SVM_model(self):
        X_train = self.data_train.drop('Survived', axis=1)
        y_train = self.data_train['Survived']
        X_test = self.data_test.drop('Survived', axis=1)
        y_test = self.data_test['Survived']
        X_train = self._prepare_features(X_train)
        X_test = self._prepare_features(X_test)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        model = SVC(probability=True)
        y_pred, scores = self._perform_and_evaluate(model, X_train, y_train, X_test, y_test)
        if scores:
            self.scores['SVM'] = scores
        return model, y_pred

    def _perform_and_evaluate(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if y_test is not None:
            return y_pred, {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
            }
        return y_pred, None

    def get_scores(self):
        return self.scores

# Usage dans le main
if __name__ == "__main__":
    # Chargement et nettoyage
    processor_train = DataPreprocessor("data_titanic/train.csv")
    processor_test = DataPreprocessor("data_titanic/test.csv")
    processor_res = pd.read_csv("data_titanic/gender_submission.csv")  

    clean_train = processor_train.clean_data()
    clean_test = processor_test.clean_data()

    # Fusion des résultats
    clean_test_with_labels = pd.merge(
        clean_test,
        processor_res[['PassengerId', 'Survived']],
        on='PassengerId'
    )


    # Modèle
    trainer = ModelTraining(clean_train, clean_test_with_labels)
    trainer.logistic_model()
    trainer.random_forest_model()
    trainer.SVM_model()

    # Résultats
    scores = trainer.get_scores()
    print(scores)
