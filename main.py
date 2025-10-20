if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    from model_training import ModelTraining
    from utils import print_model_scores
    import pandas as pd
    from pathlib import Path

    # Résolution des chemins relatifs par rapport au dossier du script
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data_titanic'
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    gender_path = data_dir / 'gender_submission.csv'

    # Chargement et nettoyage
    if not train_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {test_path}")
    if not gender_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {gender_path}")

    processor_train = DataPreprocessor(str(train_path))
    processor_test = DataPreprocessor(str(test_path))
    processor_res = pd.read_csv(str(gender_path))  
    # Nettoyage : pour le jeu de test on conserve les PassengerId (for_training=False)
    clean_train = processor_train.clean_data(for_training=True)
    clean_test = processor_test.clean_data(for_training=False)

    # Récupérer les IDs de test (préservés par clean_data avec for_training=False)
    test_ids = processor_test.get_passenger_ids()

    # Fusionner les labels fournis (gender_submission) avec les IDs de test
    if test_ids is not None:
        labels_df = processor_res[['PassengerId', 'Survived']]
        # Construire un DataFrame minimal contenant PassengerId + Survived
        clean_test_with_labels = pd.merge(
            test_ids.to_frame(name='PassengerId'),
            labels_df,
            on='PassengerId',
            how='left'
        )
        # Concaténer les features nettoyées et la colonne Survived
        clean_test_with_labels = pd.concat([clean_test.reset_index(drop=True), clean_test_with_labels['Survived'].reset_index(drop=True)], axis=1)
    else:
        # Si pas d'IDs, essayer une fusion directe si PassengerId présent
        if 'PassengerId' in clean_test.columns:
            clean_test_with_labels = pd.merge(clean_test, processor_res[['PassengerId','Survived']], on='PassengerId', how='left')
        else:
            raise RuntimeError('Impossible de récupérer les PassengerId pour le jeu de test. Vérifiez clean_data(for_training=False)')

    # Modélisation
    trainer = ModelTraining(clean_train, clean_test_with_labels)

    logistic_model_obj, logistic_predictions = trainer.logistic_model()
    rf_model_obj, rf_predictions = trainer.random_forest_model()
    svm_model_obj,svm_predictions = trainer.SVM_model()

    # Afficher les prédictions (quelques premières)
    print("Prédictions Logistic Regression :", logistic_predictions[:30])
    print("Prédictions Random Forest       :", rf_predictions[:30])
    print("Prédiction SVM        :",svm_predictions[:30])

    # Sauvegarder les prédictions dans un fichier CSV (exemple)
    output_df = clean_test.copy()
    output_df['Survived_Pred_Logistic'] = logistic_predictions
    output_df['Survived_Pred_RF'] = rf_predictions
    output_df['Survived_Pred_Svm'] = svm_predictions
    output_df.to_csv('predictions.csv', index=False)

    scores = trainer.get_scores()
    print_model_scores(scores)