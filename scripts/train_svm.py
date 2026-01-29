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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
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
    
    # Mostrar distribución de clases
    print(f"\nDistribución de clases en entrenamiento:")
    print(f"  A Tiempo (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
    print(f"  Con Demora (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test

def optimize_svm_hyperparameters(X_train, y_train, cv_folds=3, n_iter=20, random_state=42):
    """
    Optimizar hiperparámetros del modelo SVM usando RandomizedSearchCV.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        cv_folds (int): Número de folds para validación cruzada
        n_iter (int): Número de iteraciones para búsqueda aleatoria
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    print("Optimizando hiperparámetros para SVM...")
    
    # Definir distribución de parámetros para búsqueda aleatoria
    param_distributions = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
    }
    
    # Crear modelo base con class_weight='balanced' para manejar desbalance de clases
    # probability=True permite usar predict_proba() para métricas como ROC AUC
    svm = SVC(random_state=random_state, class_weight='balanced', probability=True)
    
    # Configurar validación cruzada estratificada
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Configurar RandomizedSearchCV (mucho más rápido que GridSearchCV)
    random_search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_distributions,
        n_iter=n_iter,  # Solo prueba 20 combinaciones aleatorias
        cv=cv,
        scoring='f1',
        n_jobs=1,
        random_state=random_state,
        verbose=1
    )
    
    # Ejecutar búsqueda
    print("Ejecutando búsqueda aleatoria de hiperparámetros...")
    random_search.fit(X_train, y_train)
    
    # Mostrar resultados
    print(f"Mejor puntuación F1: {random_search.best_score_:.4f}")
    print("Mejores hiperparámetros:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return random_search.best_params_, random_search.best_score_

def train_svm_model(X_train, y_train, best_params, random_state=42):
    """
    Entrenar el modelo SVM con los mejores hiperparámetros.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        best_params (dict): Mejores hiperparámetros
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        SVC: Modelo entrenado
    """
    print("Entrenando modelo SVM con mejores hiperparámetros...")
    
    # Crear modelo con mejores parámetros
    # Asegurar que class_weight='balanced' y probability=True estén presentes
    # probability=True permite usar predict_proba() para métricas como ROC AUC
    svm_model = SVC(**best_params, class_weight='balanced', probability=True)
    if 'random_state' not in best_params:
        svm_model.set_params(random_state=random_state)
    
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
    
    # Mostrar distribución de clases real
    print(f"\nDistribución de clases en conjunto de prueba:")
    print(f"  A Tiempo (0): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.2f}%)")
    print(f"  Con Demora (1): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.2f}%)")
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Mostrar distribución de predicciones
    print(f"\nDistribución de predicciones:")
    print(f"  A Tiempo (0): {(y_pred == 0).sum()} ({(y_pred == 0).mean()*100:.2f}%)")
    print(f"  Con Demora (1): {(y_pred == 1).sum()} ({(y_pred == 1).mean()*100:.2f}%)")
    
    # Calcular métricas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Obtener probabilidades para ROC AUC (requiere probability=True en el modelo)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)  # Usar probabilidades, no predicciones
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
    if len(X_train) > 10000:
        print(f"Dataset grande ({len(X_train)} muestras). Muestreando para SVM...")
        sample_size = 10000
        X_train_sample = X_train.sample(n=sample_size, random_state=42)
        y_train_sample = y_train.loc[X_train_sample.index]
        best_params, best_score = optimize_svm_hyperparameters(X_train_sample, y_train_sample)
    else:
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
