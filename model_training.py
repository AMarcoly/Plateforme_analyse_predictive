from data_preprocessing import DataPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd


class ModelTraining:
    def __init__(self,data_train,data_test):
        self.data_train = data_train
        self.data_test = data_test
        self.scores = {}
        

    def logistic_model(self):
        X_train = self.data_train.drop('Survived', axis=1)
        y_train = self.data_train['Survived']

              
        if 'Survived' in self.data_test.columns:
            X_test = self.data_test.drop('Survived', axis=1)
            y_test = self.data_test['Survived']
            evaluate = True
        else:
            X_test = self.data_test.copy()
            y_test = None
            evaluate = False

        ###############################################
        X_train = pd.get_dummies(X_train, columns=['Embarked', 'Title', 'Deck'], drop_first=True)
        X_test = pd.get_dummies(X_test, columns=['Embarked', 'Title', 'Deck'], drop_first=True)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        ###############################################

        model = LogisticRegression(max_iter=2000, solver='lbfgs')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if evaluate:
            self.scores['LogisticRegression'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            }
        else:
            print("Données de test sans label 'Survived' : évaluation impossible, prédictions uniquement.")

        return model, y_pred

    def random_forest_model(self):
        X_train = self.data_train.drop('Survived', axis=1)
        y_train = self.data_train['Survived']
        
        
        if 'Survived' in self.data_test.columns:
            X_test = self.data_test.drop('Survived', axis=1)
            y_test = self.data_test['Survived']
            evaluate = True
        else:
            X_test = self.data_test.copy()
            y_test = None
            evaluate = False

        ###############################################
        X_train = pd.get_dummies(X_train, columns=['Embarked', 'Title', 'Deck'], drop_first=True)
        X_test = pd.get_dummies(X_test, columns=['Embarked', 'Title', 'Deck'], drop_first=True)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        ###############################################

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if evaluate:
            self.scores['LogisticRegression'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            }
        else:
            print("Données de test sans label 'Survived' : évaluation impossible, prédictions uniquement.")

        return model,y_pred
    
    def SVM_model(self):
        X_train = self.data_train.drop('Survived', axis=1)
        y_train = self.data_train['Survived']

        if 'Survived' in self.data_test.columns:
            X_test = self.data_test.drop('Survived', axis=1)
            y_test = self.data_test['Survived']
            evaluate = True
        else:
            X_test = self.data_test.copy()
            y_test = None
            evaluate = False

        # Encodage one-hot identique aux autres modèles
        cols_to_encode = ['Embarked', 'Title', 'Deck']
        X_train = pd.get_dummies(X_train, columns=cols_to_encode, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cols_to_encode, drop_first=True)

        # Alignement des colonnes
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        model = SVC(probability=True)  # Proba en sortie
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if evaluate:
            self.scores['SVM'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred)
            }
        else:
            print("Données de test sans label 'Survived' : évaluation impossible, prédictions uniquement.")

        return model, y_pred

    
    def get_scores(self):
        return self.scores
        

if __name__ == "__main__":
    processor_train = DataPreprocessor("data_titanic/train.csv")
    processor_test = DataPreprocessor("data_titanic/test.csv")
    clean_data_train = processor_train.clean_data()
    clean_data_test = processor_test.clean_data()

    # model = ModelTraining(clean_data_train,clean_data_test)
