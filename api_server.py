from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import os

app = Flask(__name__)

# Chargement du modèle
MODEL_PATH = 'best_models/LogisticRegression_best_model.pkl'
model = joblib.load(MODEL_PATH)

# Récupération des vraies features du modèle
EXPECTED_FEATURES = model.feature_names_in_.tolist()

class DataPreprocessorAPI:
    @staticmethod
    def preprocess_single_input(input_data: Dict[str, Any]) -> pd.DataFrame:
        data = input_data.copy()
        df = pd.DataFrame([data])
        
        # Valeurs par défaut pour features manquantes
        if 'SibSp' not in df.columns:
            df['SibSp'] = 0
        if 'Parch' not in df.columns:
            df['Parch'] = 0
        if 'Embarked' not in df.columns:
            df['Embarked'] = 'S'
        if 'Cabin' not in df.columns:
            df['Cabin'] = 'U'
        if 'Name' not in df.columns:
            df['Name'] = 'Unknown, Mr.'
        
        # Encodage Sex
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).fillna(0).astype(int)
        
        # Features dérivées famille
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Extraction Title
        df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
        df['Title'] = df['Title'].astype(str).str.strip()
        df['Title'] = df['Title'].str.replace(r'^\s*the\s+', '', regex=True)
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
        rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
        df['Title'] = df['Title'].apply(lambda t: 'Rare' if t in rare_titles else t)
        df['Title'] = df['Title'].fillna('Mr')
        
        # Extraction Deck
        df['Deck'] = df['Cabin'].astype(str).str[0]
        df['Deck'] = df['Deck'].replace({'n': 'U', 'N': 'U'})
        df['Deck'] = df['Deck'].fillna('U')
        deck_mapping = {'T': 'U', 'F': 'Other', 'G': 'Other'}
        df['Deck'] = df['Deck'].replace(deck_mapping)
        
        # One-hot encoding avec drop_first=True (identique à l'entraînement)
        categorical_columns = ['Embarked', 'Title', 'Deck']
        # Sauvegarder les valeurs originales pour chaque catégorie afin de
        # pouvoir recréer la colonne dummy correspondante si get_dummies
        # n'en crée pas (cas d'une seule ligne où drop_first=True peut supprimer l'information).
        orig_values = {col: df[col].iloc[0] if col in df.columns else None for col in categorical_columns}

        for col in categorical_columns:
            if col in df.columns and df[col].dtype.name != 'category':
                df[col] = df[col].astype('category')

        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

        # construction manuelle liste dummies via liste principale
        expected_dummy_cols = {}
        for feat in EXPECTED_FEATURES:
            if '_' in feat:
                prefix = feat.split('_', 1)[0]
                expected_dummy_cols.setdefault(prefix, []).append(feat)

        # Init toutes les colonnes pr le dummy
        for cols in expected_dummy_cols.values():
            for c in cols:
                if c not in df.columns:
                    df[c] = 0

        # Forcer la colonne dummy correcte à 1 en fonction de la valeur originale
        for col in categorical_columns:
            val = orig_values.get(col)
            if val is None:
                continue
            val_str = str(val).strip()
            target_col = f"{col}_{val_str}"
            if target_col in df.columns:
                df[target_col] = 1
            else:
                # Cas particulier : pour Embarked
                if col == 'Deck':
                    if 'Deck_U' in df.columns and (val_str == '' or val_str.lower() == 'nan' or val_str.upper() == 'N'):
                        df['Deck_U'] = 1
                    elif 'Deck_Other' in df.columns:
                        df['Deck_Other'] = 1
        
        # Suppression colonnes inutiles
        drop_cols = [c for c in ['Name', 'Ticket', 'Cabin', 'PassengerId'] if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        return df

    @staticmethod
    def align_features(input_df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        return input_df[expected_columns]

@app.route('/')
def home():
    return jsonify({
        'message': 'Titanic Survival Prediction API',
        'endpoints': ['/health', '/model/info', '/predict', '/batch_predict']
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_type': 'LogisticRegression'
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        'model_name': 'LogisticRegression',
        'features_count': len(EXPECTED_FEATURES),
        'expected_features': EXPECTED_FEATURES
    })

@app.route('/predict', methods=['POST'])
def predict_single():
    try:
        if not request.json:
            return jsonify({'error': 'JSON data required'}), 400
        
        input_data = request.json
        print(f"Input data: {input_data}")
        
        processed_data = DataPreprocessorAPI.preprocess_single_input(input_data)
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Processed columns: {processed_data.columns.tolist()}")
        
        processed_data = DataPreprocessorAPI.align_features(processed_data, EXPECTED_FEATURES)
        print(f"Aligned data shape: {processed_data.shape}")
        print(f"Aligned values: {processed_data.values}")
        
        probability = model.predict_proba(processed_data)[0][1]
        prediction = probability > 0.5
        
        print(f"Prediction: probability={probability:.4f}, survived={prediction}")
        
        return jsonify({
            'survival_probability': round(float(probability), 4),
            'survived': bool(prediction),
            'passenger_id': input_data.get('PassengerId')
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting Flask API with model: {type(model).__name__}")
    print(f"Model features: {len(EXPECTED_FEATURES)} columns")
    print(f"Feature names: {EXPECTED_FEATURES}")
    app.run(host='127.0.0.1', port=5000, debug=True)