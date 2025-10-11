import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, file_path: str):
        self.data = pd.read_csv(file_path)
        self.cleaned_data = None

    def clean_data(self):
        # Suppression des doublons exacts
        self.data.drop_duplicates(inplace=True)

        # Remplir Embarked par la modalité la plus fréquente
        if 'Embarked' in self.data.columns:
            self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode().iloc[0])

        # Remplir Fare manquant par la médiane
        if 'Fare' in self.data.columns:
            self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())

        # Extraire Title depuis Name (ex: Mr, Mrs, Miss, Master, etc.)
        if 'Name' in self.data.columns:
            self.data['Title'] = self.data['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
            # Nettoyage des titres rares
            rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
            self.data['Title'] = self.data['Title'].replace(['Mlle','Ms'],'Miss')
            self.data['Title'] = self.data['Title'].replace(['Mme'],'Mrs')
            self.data['Title'] = self.data['Title'].apply(lambda t: 'Rare' if t in rare_titles else t)

        # Imputation d'Age : médiane par (Pclass, Sex) si Age manquant
        if 'Age' in self.data.columns:
            self.data['Age'] = self.data['Age'].astype(float)
            age_median = self.data.groupby(['Pclass','Sex'])['Age'].transform('median')
            self.data['Age'] = self.data['Age'].fillna(age_median)
            # Si certains restent NA (groupes vides), utiliser la médiane globale
            self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())

        # Traitement de Cabin -> extraire Deck (première lettre) ou 'U' pour unknown
        if 'Cabin' in self.data.columns:
            self.data['Deck'] = self.data['Cabin'].astype(str).str[0]
            self.data['Deck'] = self.data['Deck'].replace('n', 'U')
            self.data['Deck'] = self.data['Deck'].replace('N', 'U')
            self.data['Deck'] = self.data['Deck'].fillna('U')

        # Encodage simple de Sex en binaire
        if 'Sex' in self.data.columns:
            self.data['Sex'] = self.data['Sex'].map({'male':0, 'female':1}).astype(int)

        # Features dérivées : FamilySize, IsAlone
        if set(['SibSp','Parch']).issubset(self.data.columns):
            self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1
            self.data['IsAlone'] = (self.data['FamilySize'] == 1).astype(int)

        # Supprimer les colonnes peu utiles pour un modèle de base
        drop_cols = []
        for c in ['PassengerId','Name','Ticket','Cabin']:
            if c in self.data.columns:
                drop_cols.append(c)
        if drop_cols:
            self.data.drop(columns=drop_cols, inplace=True)

        # Conversion des colonnes object restantes en category
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].astype('category')

        # Stocker les données nettoyées
        self.cleaned_data = self.data.copy()
        return self.cleaned_data

    def get_data(self):
        return self.cleaned_data

if __name__ == "__main__":
    data = DataPreprocessor("data_titanic/train.csv")
    data.clean_data()
    print(data.get_data().head())
    print(data.get_data().info())
