Projet 1 : Plateforme d’analyse prédictive sur données publiques (Machine Learning)
	•	Objectif : créer un pipeline complet (acquisition, exploration, modélisation, déploiement) en Python pour prédire une variable cible d’un jeu de données open data (ex : qualité de l’air, prix immobilier).
	•	Étapes :
	•	Choisir un dataset open data large et pertinent (Kaggle, data.gouv.fr).
	•	Nettoyer et explorer les données avec Pandas.
	•	Visualiser les flux et corrélations via matplotlib/seaborn.
	•	Concevoir et tester plusieurs modèles de machine learning (scikit-learn).
	•	Optimiser (hyperparamétrage).
	•	Comparer les résultats et générer des rapports (Jupyter, Markdown).
	•	Déployer une API REST (Flask FastAPI) pour l’inférence.
	•	Documenter l’ensemble, versionner sur Git.

Choisir un jeu de données ouvert pertinent et volumineux (data.gouv.fr, Kaggle).
	•	Nettoyer et explorer les données (Python, Pandas).
	•	Visualiser les corrélations entre variables (matplotlib, seaborn).
	•	Concevoir et tester plusieurs modèles de machine learning (scikit-learn).
	•	Optimiser hyperparamètres.
	•	Comparer métriques, documenter résultats (Jupyter Notebook, Markdown).
	•	Déployer une API REST pour l’inférence (Flask ou FastAPI, Heroku).
	•	Versionner sur Git, rédiger README complet





	•	main.py : script orchestration principale (chargement, traitement, fenêtre UI ou API)
	•	data_preprocessing.py : nettoyage, exploration, visualisation (Pandas, seaborn)
	•	model_training.py : définition, entraînement, évaluation des modèles ML (scikit-learn)
	•	hyperparameter_tuning.py : optimisation des hyperparamètres (GridSearch, RandomSearch)
	•	api_server.py : API REST pour déploiement inference (Flask ou FastAPI)
	•	utils.py : fonctions utilitaires communes (gestion chemins, logger)
	•	requirements.txt : liste des dépendances Python
	•	README.md : documentation projet