"""
Script de Entrenamiento - SVM (Support Vector Machine)
======================================================

Este script entrena y optimiza un modelo SVM para predecir demoras
en entregas utilizando validación cruzada y búsqueda de hiperparámetros.

Autor: Sistema de Predicción de Demoras
Fecha: 2024
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.svm import SVC
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

def optimize_svm_hyperparameters(X_train, y_train, cv_folds=5, random_state=42):
    """
    Optimizar hiperparámetros del modelo SVM usando GridSearchCV.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        cv_folds (int): Número de folds para validación cruzada
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    print("Optimizando hiperparámetros para SVM...")
    
    # Definir parámetros para búsqueda
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    # Crear modelo base
    svm = SVC(random_state=random_state)
    
    # Configurar validación cruzada estratificada
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
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

def train_svm_model(X_train, y_train, best_params):
    """
    Entrenar el modelo SVM con los mejores hiperparámetros.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        best_params (dict): Mejores hiperparámetros
        
    Returns:
        SVC: Modelo entrenado
    """
    print("Entrenando modelo SVM con mejores hiperparámetros...")
    
    # Crear modelo con mejores parámetros
    svm_model = SVC(**best_params)
    
    # Entrenar modelo
    svm_model.fit(X_train, y_train)
    
    print("Modelo SVM entrenado exitosamente")
    return svm_model

def evaluate_svm_model(model, X_test, y_test):
    """
    Evaluar el modelo SVM en el conjunto de prueba.
    
    Args:
        model: Modelo entrenado
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        
    Returns:
        dict: Métricas de evaluación
    """
    print("Evaluando modelo SVM...")
    
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

def save_svm_model(model, best_params, metrics, model_dir='models', results_dir='data/results'):
    """
    Guardar el modelo SVM y sus resultados.
    
    Args:
        model: Modelo entrenado
        best_params (dict): Mejores hiperparámetros
        metrics (dict): Métricas de evaluación
        model_dir (str): Directorio para guardar el modelo
        results_dir (str): Directorio para guardar resultados
    """
    print("Guardando modelo SVM y resultados...")
    
    # Crear directorios si no existen
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar modelo
    joblib.dump(model, f'{model_dir}/svm_model.pkl')
    print(f"Modelo guardado en: {model_dir}/svm_model.pkl")
    
    # Guardar hiperparámetros
    with open(f'{results_dir}/svm_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Hiperparámetros guardados en: {results_dir}/svm_params.json")
    
    # Guardar métricas
    with open(f'{results_dir}/svm_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas guardadas en: {results_dir}/svm_metrics.json")

def main():
    """
    Función principal para entrenar el modelo SVM.
    """
    print("="*60)
    print("ENTRENAMIENTO DEL MODELO SVM")
    print("="*60)
    
    # 1. Cargar datos procesados
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # 2. Optimizar hiperparámetros
    best_params, best_score = optimize_svm_hyperparameters(X_train, y_train)
    
    # 3. Entrenar modelo con mejores parámetros
    svm_model = train_svm_model(X_train, y_train, best_params)
    
    # 4. Evaluar modelo
    metrics = evaluate_svm_model(svm_model, X_test, y_test)
    
    # 5. Guardar modelo y resultados
    save_svm_model(svm_model, best_params, metrics)
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO SVM COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return svm_model, best_params, metrics

if __name__ == "__main__":
    model, params, metrics = main()
