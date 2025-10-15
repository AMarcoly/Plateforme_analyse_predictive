import pandas as pd

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

    def drop_duplicates(self):
        self.data.drop_duplicates(inplace=True)

    def impute_embarked(self):
        if 'Embarked' in self.data.columns:
            self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode().iloc[0])

    def impute_fare(self):
        if 'Fare' in self.data.columns:
            self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())

    def extract_title(self):
        if 'Name' in self.data.columns:
            self.data['Title'] = self.data['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
            rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
            self.data['Title'] = self.data['Title'].replace(['Mlle','Ms'],'Miss')
            self.data['Title'] = self.data['Title'].replace(['Mme'],'Mrs')
            self.data['Title'] = self.data['Title'].apply(lambda t: 'Rare' if t in rare_titles else t)

    def impute_age(self):
        if 'Age' in self.data.columns:
            self.data['Age'] = self.data['Age'].astype(float)
            age_median = self.data.groupby(['Pclass','Sex'])['Age'].transform('median')
            self.data['Age'] = self.data['Age'].fillna(age_median)
            self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())

    def extract_deck(self):
        if 'Cabin' in self.data.columns:
            self.data['Deck'] = self.data['Cabin'].astype(str).str[0]
            self.data['Deck'] = self.data['Deck'].replace('n', 'U')
            self.data['Deck'] = self.data['Deck'].replace('N', 'U')
            self.data['Deck'] = self.data['Deck'].fillna('U')

    def encode_sex(self):
        if 'Sex' in self.data.columns:
            self.data['Sex'] = self.data['Sex'].map({'male':0, 'female':1}).astype(int)

    def create_family_features(self):
        if set(['SibSp','Parch']).issubset(self.data.columns):
            self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1
            self.data['IsAlone'] = (self.data['FamilySize'] == 1).astype(int)

    def drop_useless_columns(self):
        drop_cols = [c for c in ['PassengerId','Name','Ticket','Cabin'] if c in self.data.columns]
        if drop_cols:
            self.data.drop(columns=drop_cols, inplace=True)

    def convert_object_to_category(self):
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].astype('category')

    def clean_data(self):
        self.drop_duplicates()
        self.impute_embarked()
        self.impute_fare()
        self.extract_title()
        self.impute_age()
        self.extract_deck()
        self.encode_sex()
        self.create_family_features()
        self.drop_useless_columns()
        self.convert_object_to_category()
        self.cleaned_data = self.data.copy()
        return self.cleaned_data

    def get_data(self):
        return self.cleaned_data

if __name__ == "__main__":
    processor = DataPreprocessor("data_titanic/train.csv")
    processor.clean_data()
    print(processor.get_data().head())
    print(processor.get_data().info())
