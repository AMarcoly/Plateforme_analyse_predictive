import pandas as pd
import numpy as np
import seaborn as sns

class DataPreprocessor:
    """
    Classe pour le prétraitement des données du jeu Titanic.
    """
    def __init__(self, file_path: str):
        # Chargement des données brutes depuis le fichier CSV
        self.data = pd.read_csv(file_path)
        # Initialisation de l'attribut pour stocker les données nettoyées
        self.cleaned_data = None

    def clean_data(self,for_training=True):
        # Création d'une copie des données brutes pour préserver l'original
        data_clean = self.data.copy()
        passenger_ids = None

        # Suppression des doublons exacts (lignes identiques)
        data_clean.drop_duplicates(inplace=True)

        # Imputation des valeurs manquantes pour 'Embarked' avec le mode (valeur la plus fréquente)
        if 'Embarked' in data_clean.columns:
            # Récupère la valeur la plus fréquente, utilise 'S' si aucun mode n'existe
            embarked_mode = data_clean['Embarked'].mode()
            mode_value = embarked_mode.iloc[0] if not embarked_mode.empty else 'S'
            # Remplace les NaN par la valeur du mode
            data_clean['Embarked'] = data_clean['Embarked'].fillna(mode_value)

        # Imputation des valeurs manquantes pour 'Fare' avec la médiane
        if 'Fare' in data_clean.columns:
            # Calcule la médiane des tarifs et remplace les NaN
            fare_median = data_clean['Fare'].median()
            data_clean['Fare'] = data_clean['Fare'].fillna(fare_median)

        # Extraction et normalisation du titre depuis la colonne 'Name'
        if 'Name' in data_clean.columns:
            # Regex pour extraire le titre entre la virgule et le point (ex: "Mr", "Mrs")
            data_clean['Title'] = data_clean['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
            # Suppression les espaces superflus
            data_clean['Title'] = data_clean['Title'].astype(str).str.strip()
            # Nettoyage préfixes 
            data_clean['Title'] = data_clean['Title'].str.replace(r'^\s*the\s+', '', regex=True)
            # Harmonisation des titres équivalents
            data_clean['Title'] = data_clean['Title'].replace(['Mlle', 'Ms'], 'Miss')  # Mademoiselle -> Miss
            data_clean['Title'] = data_clean['Title'].replace(['Mme'], 'Mrs')  # Madame -> Mrs
            # Regroupement des titres rares dans une catégorie "Rare"
            rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
            data_clean['Title'] = data_clean['Title'].apply(lambda t: 'Rare' if t in rare_titles else t)

        # Imputation de l'âge avec la médiane par groupe (Classe, Sexe)
        if 'Age' in data_clean.columns:
            # Conversion en float pour garantir le type numérique
            data_clean['Age'] = data_clean['Age'].astype(float)
            # Vérifie que les colonnes de regroupement existent
            if set(['Pclass','Sex']).issubset(data_clean.columns):
                # Calcule la médiane d'âge pour chaque combinaison Pclass/Sexe
                age_median = data_clean.groupby(['Pclass','Sex'])['Age'].transform('median')
                # Remplace les NaN par la médiane de leur groupe
                data_clean['Age'] = data_clean['Age'].fillna(age_median)
            # Imputation finale avec la médiane globale si des NaN persistent
            data_clean['Age'] = data_clean['Age'].fillna(data_clean['Age'].median())

        # Extraction du pont (Deck) depuis la première lettre de 'Cabin'
        if 'Cabin' in data_clean.columns:
            data_clean['Deck'] = data_clean['Cabin'].astype(str).str[0]
            data_clean['Deck'] = data_clean['Deck'].replace({'n': 'U', 'N': 'U'})
            data_clean['Deck'] = data_clean['Deck'].fillna('U')
            # Regroupement des ponts rares : T avec U, F et G avec "Other"
            deck_mapping = {'T': 'U', 'F': 'Other', 'G': 'Other'}
            data_clean['Deck'] = data_clean['Deck'].replace(deck_mapping)

        #  Encodage binaire de la variable Sex (male: 0, female: 1)
        if 'Sex' in data_clean.columns:
            data_clean['Sex'] = data_clean['Sex'].map({'male': 0, 'female': 1}).astype(int)

        #  Création de variables dérivées sur la composition familiale
        if set(['SibSp','Parch']).issubset(data_clean.columns):
            # Taille de la famille = frères/soeurs + parents/enfants + le passager lui-même
            data_clean['FamilySize'] = data_clean['SibSp'] + data_clean['Parch'] + 1
            # Indicateur binaire pour les passagers voyageant seuls
            data_clean['IsAlone'] = (data_clean['FamilySize'] == 1).astype(int)

        #  Suppression des colonnes peu informatives pour la modélisation
        drop_cols = [c for c in [ 'Name','Ticket','Cabin'] if c in data_clean.columns]
        if drop_cols:
            data_clean.drop(columns=drop_cols, inplace=True)

        #  Conversion des colonnes texte restantes en type catégoriel
        categorical_columns = ['Embarked', 'Title', 'Deck']
    
        for col in categorical_columns:
            if col in data_clean.columns:
                # Conversion en category si ce n'est pas déjà fait
                if data_clean[col].dtype.name != 'category':
                    data_clean[col] = data_clean[col].astype('category')
        
        # Application du one-hot encoding
        data_clean = pd.get_dummies(data_clean, columns=categorical_columns, drop_first=True,dtype=int)

        # afectation
        self.cleaned_data = data_clean

        # Gestion ID
        if for_training and 'PassengerId' in data_clean.columns:
            self.passenger_ids = data_clean['PassengerId'].copy()  # Sauvegarde
            data_clean.drop(columns=['PassengerId'], inplace=True)
        else:
            self.passenger_ids = None
            
        return data_clean


    def get_data(self):
        # Retourne les données nettoyées (None si clean_data() n'a pas été appelé)
        return self.cleaned_data
    
    def get_passenger_ids(self):
        return self.passenger_ids

if __name__ == "__main__":
    # Test
    data = DataPreprocessor("data_titanic/train.csv")
    data.clean_data(False)
    print(data.get_data().head())
    print(data.get_data().info())