"""
Script de Entrenamiento - XGBoost (Extreme Gradient Boosting)
==============================================================

Este script entrena y optimiza un modelo XGBoost para predecir demoras
en entregas utilizando validación cruzada y búsqueda de hiperparámetros.

Autor: Sistema de Predicción de Demoras
Fecha: 2024
"""

import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(data_dir='data/processed'):
    """
    Cargar los datos procesados.
    
    Args:
        data_dir (str): Directorio con los datos procesados
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Cargando datos procesados...")
    
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').iloc[:, 0]  # Obtener serie, no DataFrame
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').iloc[:, 0]
    
    print(f"Conjunto de entrenamiento: {X_train.shape}")
    print(f"Conjunto de prueba: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def optimize_xgboost_hyperparameters(X_train, y_train, cv_folds=3, random_state=42):
    """
    Optimizar hiperparámetros del modelo XGBoost usando GridSearchCV.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        cv_folds (int): Número de folds para validación cruzada
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    print("Optimizando hiperparámetros para XGBoost...")
    
    # Definir parámetros para búsqueda
    param_grid = {
        'n_estimators': [200],
        'max_depth': [5, 7],
        'learning_rate': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    
    # Crear modelo base
    xgb_model = xgb.XGBClassifier(
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Configurar validación cruzada estratificada
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=1,
        verbose=1
    )
    
    # Ejecutar búsqueda
    print("Ejecutando búsqueda de hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    # Mostrar resultados
    print(f"Mejor puntuación F1: {grid_search.best_score_:.4f}")
    print("Mejores hiperparámetros:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return grid_search.best_params_, grid_search.best_score_

def train_xgboost_model(X_train, y_train, best_params):
    """
    Entrenar el modelo XGBoost con los mejores hiperparámetros.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        best_params (dict): Mejores hiperparámetros
        
    Returns:
        XGBClassifier: Modelo entrenado
    """
    print("Entrenando modelo XGBoost con mejores hiperparámetros...")
    
    # Crear modelo con mejores parámetros
    xgb_model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Entrenar modelo
    xgb_model.fit(X_train, y_train)
    
    print("Modelo XGBoost entrenado exitosamente")
    return xgb_model

def evaluate_xgboost_model(model, X_test, y_test):
    """
    Evaluar el modelo XGBoost en el conjunto de prueba.
    
    Args:
        model: Modelo entrenado
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        
    Returns:
        dict: Métricas de evaluación
    """
    print("Evaluando modelo XGBoost...")
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }
    
    # Mostrar métricas
    print("\nMÉTRICAS DE EVALUACIÓN:")
    print("="*30)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Mostrar reporte de clasificación
    print("\nREPORTE DE CLASIFICACIÓN:")
    print("="*30)
    print(classification_report(y_test, y_pred, target_names=['A Tiempo', 'Con Demora']))
    
    # Mostrar matriz de confusión
    print("\nMATRIZ DE CONFUSIÓN:")
    print("="*25)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicción")
    print(f"                 A Tiempo  Con Demora")
    print(f"Real A Tiempo     {cm[0,0]:8d}  {cm[0,1]:10d}")
    print(f"Real Con Demora   {cm[1,0]:8d}  {cm[1,1]:10d}")
    
    return metrics

def get_feature_importance(model, feature_names):
    """
    Obtener importancia de características del modelo XGBoost.
    
    Args:
        model: Modelo entrenado
        feature_names: Lista de nombres de características
        
    Returns:
        pd.DataFrame: DataFrame con importancia de características
    """
    print("Obteniendo importancia de características...")
    
    # Obtener importancia de características
    importance_scores = model.feature_importances_
    
    # Crear DataFrame con importancia
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print("Top 10 características más importantes:")
    print(feature_importance.head(10))
    
    return feature_importance

def save_xgboost_model(model, best_params, metrics, feature_importance, 
                      model_dir='models', results_dir='data/results'):
    """
    Guardar el modelo XGBoost y sus resultados.
    
    Args:
        model: Modelo entrenado
        best_params (dict): Mejores hiperparámetros
        metrics (dict): Métricas de evaluación
        feature_importance (pd.DataFrame): Importancia de características
        model_dir (str): Directorio para guardar el modelo
        results_dir (str): Directorio para guardar resultados
    """
    print("Guardando modelo XGBoost y resultados...")
    
    # Crear directorios si no existen
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar modelo
    joblib.dump(model, f'{model_dir}/xgboost_model.pkl')
    print(f"Modelo guardado en: {model_dir}/xgboost_model.pkl")
    
    # Guardar hiperparámetros
    with open(f'{results_dir}/xgboost_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Hiperparámetros guardados en: {results_dir}/xgboost_params.json")
    
    # Guardar métricas
    with open(f'{results_dir}/xgboost_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas guardadas en: {results_dir}/xgboost_metrics.json")
    
    # Guardar importancia de características
    feature_importance.to_csv(f'{results_dir}/xgboost_feature_importance.csv', index=False)
    print(f"Importancia de características guardada en: {results_dir}/xgboost_feature_importance.csv")

def main():
    """
    Función principal para entrenar el modelo XGBoost.
    """
    print("="*60)
    print("ENTRENAMIENTO DEL MODELO XGBOOST")
    print("="*60)
    
    # 1. Cargar datos procesados
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # 2. Optimizar hiperparámetros
    best_params, best_score = optimize_xgboost_hyperparameters(X_train, y_train)
    
    # 3. Entrenar modelo con mejores parámetros
    xgb_model = train_xgboost_model(X_train, y_train, best_params)
    
    # 4. Evaluar modelo
    metrics = evaluate_xgboost_model(xgb_model, X_test, y_test)
    
    # 5. Obtener importancia de características
    feature_names = X_train.columns.tolist()
    feature_importance = get_feature_importance(xgb_model, feature_names)
    
    # 6. Guardar modelo y resultados
    save_xgboost_model(xgb_model, best_params, metrics, feature_importance)
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO XGBOOST COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return xgb_model, best_params, metrics, feature_importance

if __name__ == "__main__":
    model, params, metrics, feature_importance = main()
