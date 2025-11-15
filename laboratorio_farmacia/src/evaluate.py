import json
import pandas as pd

def evaluate_model():
    # Cargar métricas
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Crear resumen
    summary = {
        'best_model': metrics['best_model'],
        'best_r2': metrics['best_r2'],
        'all_models': {}
    }
    
    for model_name, model_metrics in metrics.items():
        if model_name not in ['best_model', 'best_r2']:
            summary['all_models'][model_name] = model_metrics
    
    # Guardar resumen
    with open('evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Evaluación completada!")
    print(f"Modelo ganador: {summary['best_model']}")
    print(f"R2 score: {summary['best_r2']:.4f}")
    
    # Mostrar comparación de modelos
    print("\n--- COMPARACIÓN DE MODELOS ---")
    for model_name, model_metrics in summary['all_models'].items():
        print(f"{model_name}:")
        print(f"  - R2: {model_metrics['R2']:.4f}")
        print(f"  - MAE: {model_metrics['MAE']:.2f}")
        print(f"  - MSE: {model_metrics['MSE']:.2f}")

if __name__ == "__main__":
    evaluate_model()
    