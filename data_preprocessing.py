import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

class DataPreprocessor:
    """
    Classe pour le prétraitement des données du jeu Titanic.
    Args:
        file_path (str): Chemin vers le fichier CSV contenant les données.
    Attributs:
        data (pd.DataFrame): Données brutes chargées depuis le fichier.
        cleaned_data (pd.DataFrame): Données nettoyées après prétraitement.
    Méthodes:
        clean_data():
            Effectue les étapes de nettoyage et d'ingénierie des variables :
                - Suppression des doublons exacts.
                - Imputation des valeurs manquantes pour 'Embarked' (mode), 'Fare' (médiane), et 'Age' (médiane par groupe).
                - Extraction et normalisation du titre depuis 'Name'.
                - Extraction du pont ('Deck') depuis 'Cabin', gestion des valeurs inconnues.
                - Encodage binaire de la variable 'Sex'.
                - Création des variables dérivées 'FamilySize' et 'IsAlone'.
                - Suppression des colonnes peu informatives pour la modélisation.
                - Conversion des colonnes de type objet en catégorie.
            Retourne le DataFrame nettoyé.
        get_data():
            Retourne les données nettoyées (cleaned_data).
    """
    def __init__(self, file_path: str):
        self.data = pd.read_csv(file_path)
        self.cleaned_data = None

    def clean_data(self):
        # Suppression des doublons exacts
        self.data.drop_duplicates(inplace=True)

        # Embarked -> mode
        if 'Embarked' in self.data.columns:
            self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode().iloc[0])

        # Fare -> médiane
        if 'Fare' in self.data.columns:
            self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())

        # Title depuis Name
        if 'Name' in self.data.columns:
            self.data['Title'] = self.data['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
            self.data['Title'] = self.data['Title'].astype(str).str.strip()
            # enlever 'the ' au début si présent
            self.data['Title'] = self.data['Title'].str.replace(r'^\s*the\s+', '', regex=True)
            # Harmonisation
            self.data['Title'] = self.data['Title'].replace(['Mlle', 'Ms'], 'Miss')
            self.data['Title'] = self.data['Title'].replace(['Mme'], 'Mrs')
            rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
            self.data['Title'] = self.data['Title'].apply(lambda t: 'Rare' if t in rare_titles else t)

        # Imputation Age par médiane groupée (Pclass, Sex)
        if 'Age' in self.data.columns:
            self.data['Age'] = self.data['Age'].astype(float)
            if set(['Pclass','Sex']).issubset(self.data.columns):
                age_median = self.data.groupby(['Pclass','Sex'])['Age'].transform('median')
                self.data['Age'] = self.data['Age'].fillna(age_median)
            self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())

        # Deck depuis Cabin
        if 'Cabin' in self.data.columns:
            self.data['Deck'] = self.data['Cabin'].astype(str).str[0]
            self.data['Deck'] = self.data['Deck'].replace({'n': 'U', 'N': 'U'})
            self.data['Deck'] = self.data['Deck'].fillna('U')
            # Regrouper T très rare en U
            self.data['Deck'] = self.data['Deck'].where(~self.data['Deck'].isin(['T']), 'U')

        # Encodage binaire de Sex
        if 'Sex' in self.data.columns:
            self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1}).astype(int)

        # Features dérivées
        if set(['SibSp','Parch']).issubset(self.data.columns):
            self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1
            self.data['IsAlone'] = (self.data['FamilySize'] == 1).astype(int)

        # Drop colonnes peu utiles
        drop_cols = [c for c in ['Name','Ticket','Cabin'] if c in self.data.columns]
        if drop_cols:
            self.data.drop(columns=drop_cols, inplace=True)

        # Convertir objets restants en catégories
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].astype('category')

        # pipeline sklearn pour la gestion de l’encodage.
        self.cleaned_data = self.data.copy()
        return self.cleaned_data

    def get_data(self):
        return self.cleaned_data
    
    # Data visualization 


if __name__ == "__main__":
    data = DataPreprocessor("data_titanic/train.csv")
    data.clean_data()
    print(data.get_data().head())
    print(data.get_data().info())
