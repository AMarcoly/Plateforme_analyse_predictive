import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import DataPreprocessor

class DataVisualizer:
    """
    Classe pour la visualisation des données du jeu Titanic.
    """

    def __init__(self, data):
        self.data = data

    # analyse univariée
    def plot_age_distribution(self):
        """
        """
        plt.figure(figsize=(8,5))
        sns.histplot(self.data['Age'].dropna(),bins = 30, kde = True)
        plt.title('Distribution des âges')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

    def plot_sexe_repartition(self):
        """Affiche la repartition des sexes"""
        counts = self.data['Sex'].value_counts()
        labels = counts.index.map({0: "Male", 1: "Female"})
        sizes = counts.values
        colors = sns.color_palette("bright") 
        plt.title('Répartition des passagers par sexe')
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.axis('equal')
        plt.show()

    def plot_fare_distribution(self):
        """Affiche la distribution des tarifs."""
        plt.figure(figsize=(10,6))
        sns.histplot(self.data['Fare'].dropna(), bins=30, kde=True)
        plt.title('Distribution des Tarifs')
        plt.xlabel('Fare')
        plt.ylabel('Count')
        plt.show()

    def plot_survival_distribution(self):
        """Affiche la distribution des survivants."""
        plt.figure(figsize=(8,5))
        sns.countplot(x='Survived', data=self.data)
        plt.title('Distribution des Survivants')
        plt.xlabel('Survived (0 = No, 1 = Yes)')
        plt.ylabel('Count')
        plt.show()


    # analyse bivariée
    def plot_survival_by_class(self):
        """Affiche la survie par classe de billet."""
        plt.figure(figsize=(10,6))
        sns.countplot(x='Pclass', hue='Survived', data=self.data)
        plt.title('Survie par Classe de Billet')
        plt.xlabel('Pclass')
        plt.ylabel('Count')
        plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
        plt.show()

    def plot_survival_by_sex(self):
        """Affiche la survie par sexe"""
        plt.figure(figsize=(10,6))
        sns.countplot(x='Sex', hue='Survived', data=self.data)
        plt.title('Survie par Sexe')
        plt.xlabel('Sex')
        plt.ylabel('Count')
        plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
        plt.show()

    def plot_survival_compo_family(self):
        """Affiche la survie selon la composition de la famille"""
        plt.figure(figsize=(10,6))
        sns.countplot(x='FamilySize', hue='Survived', data=self.data)
        plt.title('Survie selon composition familiale')
        plt.xlabel('FamilySize')
        plt.ylabel('Count')
        plt.legend(title='FamilySize', loc='upper right', labels=['No', 'Yes'])
        plt.show()
     
    def plot_correlation_heatmap(self):
        """Affiche une carte de chaleur des corrélations entre les variables numériques."""
        plt.figure(figsize=(12,8))
        numeric_df = self.data.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Carte de Chaleur des Corrélations')
        plt.show()


if __name__ == "__main__":
    processor = DataPreprocessor("data_titanic/train.csv")
    cleaned_data = processor.clean_data()
    
    visualizer = DataVisualizer(cleaned_data)
    # visualizer.plot_age_distribution()
    # visualizer.plot_sexe_repartition()
    # visualizer.plot_survival_distribution() 
    # visualizer.plot_fare_distribution()
    # visualizer.plot_survival_by_class()
    # visualizer.plot_survival_by_sex()
    # visualizer.plot_survival_compo_family()
    visualizer.plot_correlation_heatmap()