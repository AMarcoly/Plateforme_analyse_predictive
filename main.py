if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    from model_training import ModelTraining

    # Prétraitement
    processor_train = DataPreprocessor("data_titanic/train.csv")
    processor_test = DataPreprocessor("data_titanic/test.csv")
    clean_data_train = processor_train.clean_data()
    clean_data_test = processor_test.clean_data()

    

    # Modélisation
    trainer = ModelTraining(clean_data_train, clean_data_test)

    logistic_model_obj, logistic_predictions = trainer.logistic_model()
    rf_model_obj, rf_predictions = trainer.random_forest_model()
    svm_model_obj,svm_predictions = trainer.SVM_model()

    # Afficher les prédictions (quelques premières)
    print("Prédictions Logistic Regression :", logistic_predictions[:30])
    print("Prédictions Random Forest       :", rf_predictions[:30])
    print("Prédiction Random Forest        :",svm_predictions[:30])

    # Sauvegarder les prédictions dans un fichier CSV (exemple)
    output_df = clean_data_test.copy()
    output_df['Survived_Pred_Logistic'] = logistic_predictions
    output_df['Survived_Pred_RF'] = rf_predictions
    output_df['Survived_Pred_Svm'] = svm_predictions
    output_df.to_csv('predictions.csv', index=False)