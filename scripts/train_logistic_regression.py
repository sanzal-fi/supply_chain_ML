"""
Script de Entrenamiento - Regresión Logística
==============================================

Este script entrena y optimiza un modelo de Regresión Logística para predecir demoras
en entregas utilizando validación cruzada y búsqueda de hiperparámetros.

Autor: Sistema de Predicción de Demoras
Fecha: 2024
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
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
    
    # Verificar y manejar valores NaN
    nan_train = X_train.isna().sum().sum()
    nan_test = X_test.isna().sum().sum()
    
    if nan_train > 0 or nan_test > 0:
        print(f"\n⚠️  Valores NaN detectados: {nan_train} en entrenamiento, {nan_test} en prueba")
        print("Imputando valores faltantes con la mediana...")
        
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        print("Valores NaN imputados exitosamente.")
    
    return X_train, X_test, y_train, y_test

def optimize_logistic_regression_hyperparameters(X_train, y_train, cv_folds=3, random_state=42):
    """
    Optimizar hiperparámetros del modelo de Regresión Logística usando GridSearchCV.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        cv_folds (int): Número de folds para validación cruzada
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        dict: Mejores hiperparámetros encontrados
    """
    print("Optimizando hiperparámetros para Regresión Logística...")
    
    # Definir parámetros para búsqueda
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [200, 500, 1000]
    }
    
    # Crear modelo base con class_weight='balanced' para manejar desbalance de clases
    log_reg = LogisticRegression(random_state=random_state, class_weight='balanced')
    
    # Configurar validación cruzada estratificada
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=log_reg,
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

def train_logistic_regression_model(X_train, y_train, best_params, random_state=42):
    """
    Entrenar el modelo de Regresión Logística con los mejores hiperparámetros.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        best_params (dict): Mejores hiperparámetros
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        LogisticRegression: Modelo entrenado
    """
    print("Entrenando modelo de Regresión Logística con mejores hiperparámetros...")
    
    # Crear modelo con mejores parámetros
    log_reg_model = LogisticRegression(
        **best_params,
        random_state=random_state,
        class_weight='balanced'
    )
    
    # Entrenar modelo
    log_reg_model.fit(X_train, y_train)
    
    print("Modelo de Regresión Logística entrenado exitosamente")
    return log_reg_model

def evaluate_logistic_regression_model(model, X_test, y_test):
    """
    Evaluar el modelo de Regresión Logística en el conjunto de prueba.
    
    Args:
        model: Modelo entrenado
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        
    Returns:
        dict: Métricas de evaluación
    """
    print("Evaluando modelo de Regresión Logística...")
    
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
    
    # Obtener probabilidades para ROC AUC
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
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

def get_feature_coefficients(model, feature_names):
    """
    Obtener coeficientes de características del modelo de Regresión Logística.
    
    Args:
        model: Modelo entrenado
        feature_names: Lista de nombres de características
        
    Returns:
        pd.DataFrame: DataFrame con coeficientes de características
    """
    print("Obteniendo coeficientes de características...")
    
    # Obtener coeficientes
    coefficients = model.coef_[0]
    
    # Crear DataFrame con coeficientes (ordenamos por valor absoluto)
    feature_coef = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Top 10 características más influyentes:")
    print(feature_coef[['feature', 'coefficient']].head(10))
    
    return feature_coef

def save_logistic_regression_model(model, best_params, metrics, feature_coef, 
                                   model_dir='models', results_dir='data/results'):
    """
    Guardar el modelo de Regresión Logística y sus resultados.
    
    Args:
        model: Modelo entrenado
        best_params (dict): Mejores hiperparámetros
        metrics (dict): Métricas de evaluación
        feature_coef (pd.DataFrame): Coeficientes de características
        model_dir (str): Directorio para guardar el modelo
        results_dir (str): Directorio para guardar resultados
    """
    print("Guardando modelo de Regresión Logística y resultados...")
    
    # Crear directorios si no existen
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar modelo
    joblib.dump(model, f'{model_dir}/logistic_regression_model.pkl')
    print(f"Modelo guardado en: {model_dir}/logistic_regression_model.pkl")
    
    # Guardar hiperparámetros
    with open(f'{results_dir}/logistic_regression_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Hiperparámetros guardados en: {results_dir}/logistic_regression_params.json")
    
    # Guardar métricas
    with open(f'{results_dir}/logistic_regression_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas guardadas en: {results_dir}/logistic_regression_metrics.json")
    
    # Guardar coeficientes de características
    feature_coef.to_csv(f'{results_dir}/logistic_regression_coefficients.csv', index=False)
    print(f"Coeficientes guardados en: {results_dir}/logistic_regression_coefficients.csv")

def main():
    """
    Función principal para entrenar el modelo de Regresión Logística.
    """
    print("="*60)
    print("ENTRENAMIENTO DEL MODELO DE REGRESIÓN LOGÍSTICA")
    print("="*60)
    
    # 1. Cargar datos procesados
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # 2. Optimizar hiperparámetros
    best_params, best_score = optimize_logistic_regression_hyperparameters(X_train, y_train)
    
    # 3. Entrenar modelo con mejores parámetros
    log_reg_model = train_logistic_regression_model(X_train, y_train, best_params)
    
    # 4. Evaluar modelo
    metrics = evaluate_logistic_regression_model(log_reg_model, X_test, y_test)
    
    # 5. Obtener coeficientes de características
    feature_names = X_train.columns.tolist()
    feature_coef = get_feature_coefficients(log_reg_model, feature_names)
    
    # 6. Guardar modelo y resultados
    save_logistic_regression_model(log_reg_model, best_params, metrics, feature_coef)
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO REGRESIÓN LOGÍSTICA COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return log_reg_model, best_params, metrics, feature_coef

if __name__ == "__main__":
    model, params, metrics, feature_coef = main()
