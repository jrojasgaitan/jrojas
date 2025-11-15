import pandas as pd
import yaml
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def load_config():
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_models():
    config = load_config()
    
    # Cargar datos preprocesados
    X_processed, y = joblib.load('models/processed_data.joblib')
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # Entrenar modelos
    models_config = config['models']
    results = {}
    best_score = float('-inf')
    best_model = None
    best_model_name = None
    
    for model_name, params in models_config.items():
        print(f"Entrenando {model_name}...")
        
        if model_name == "LinearRegression":
            model = LinearRegression(**params)
        elif model_name == "RandomForest":
            model = RandomForestRegressor(random_state=42, **params)
        elif model_name == "GradientBoosting":
            model = GradientBoostingRegressor(random_state=42, **params)
        else:
            continue
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            'MAE': float(mae),
            'MSE': float(mse),
            'R2': float(r2)
        }
        
        # Verificar si es el mejor modelo
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = model_name
    
    # Guardar mejor modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.joblib')
    
    # Guardar resultados
    results['best_model'] = best_model_name
    results['best_r2'] = float(best_score)
    
    with open('metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Mejor modelo: {best_model_name} con R2: {best_score:.4f}")

if __name__ == "__main__":
    train_models()