import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def load_config():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def preprocess_data():
    config = load_config()
    
    # Cargar datos
    df = pd.read_csv(config['data']['file'])
    print(f"Datos originales: {df.shape}")
    
    # Limpiar datos
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"Datos después de limpieza: {df.shape}")
    
    # Separar características y target
    X = df.drop(columns=[config['data']['target']])
    y = df[config['data']['target']]
    
    # Preprocesamiento
    numeric_features = config['preprocessing']['numeric_features']
    categorical_features = config['preprocessing']['categorical_features']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Guardar preprocesador y datos
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump((X_processed, y), 'models/processed_data.joblib')
    
    print("Preprocesamiento completado!")

if __name__ == "__main__":
    preprocess_data()