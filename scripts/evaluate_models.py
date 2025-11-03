"""
Script de Evaluación y Comparación de Modelos
==============================================

Este script evalúa y compara los tres modelos entrenados (KNN, SVM, XGBoost)
generando métricas, visualizaciones y reportes comparativos.

Autor: Sistema de Predicción de Demoras
Fecha: 2024
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           roc_curve, classification_report)
import os
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_models_and_data():
    """
    Cargar modelos entrenados y datos de prueba.
    
    Returns:
        tuple: (models, X_test, y_test)
    """
    print("Cargando modelos y datos de prueba...")
    
    # Cargar datos de prueba
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').iloc[:, 0]
    
    # Cargar modelos
    models = {}
    model_names = ['knn', 'svm', 'xgboost']
    
    for name in model_names:
        try:
            models[name] = joblib.load(f'models/{name}_model.pkl')
            print(f"Modelo {name.upper()} cargado exitosamente")
        except FileNotFoundError:
            print(f"Advertencia: No se encontró el modelo {name.upper()}")
    
    print(f"Modelos cargados: {list(models.keys())}")
    return models, X_test, y_test

def evaluate_single_model(model, model_name, X_test, y_test):
    """
    Evaluar un modelo individual.
    
    Args:
        model: Modelo entrenado
        model_name (str): Nombre del modelo
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        
    Returns:
        dict: Métricas de evaluación
    """
    print(f"\nEvaluando modelo {model_name.upper()}...")
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }
    
    # Obtener probabilidades para curva ROC (si el modelo las soporta)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics['has_probabilities'] = True
    except:
        y_proba = None
        metrics['has_probabilities'] = False
    
    return metrics, y_pred, y_proba

def create_comparison_table(models, X_test, y_test):
    """
    Crear tabla comparativa de métricas.
    
    Args:
        models (dict): Diccionario con modelos
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        
    Returns:
        pd.DataFrame: Tabla comparativa
    """
    print("Creando tabla comparativa de modelos...")
    
    results = []
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        metrics, y_pred, y_proba = evaluate_single_model(model, name, X_test, y_test)
        results.append(metrics)
        predictions[name] = y_pred
        if y_proba is not None:
            probabilities[name] = y_proba
    
    # Crear DataFrame comparativo
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model')
    
    # Redondear métricas
    numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    for col in numeric_cols:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].round(4)
    
    print("\nTABLA COMPARATIVA DE MÉTRICAS:")
    print("="*50)
    print(comparison_df)
    
    return comparison_df, predictions, probabilities

def plot_confusion_matrices(models, X_test, y_test, predictions):
    """
    Crear gráficos de matrices de confusión.
    
    Args:
        models (dict): Diccionario con modelos
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        predictions (dict): Diccionario con predicciones
    """
    print("Generando matrices de confusión...")
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = predictions[name]
        cm = confusion_matrix(y_test, y_pred)
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['A Tiempo', 'Con Demora'],
                   yticklabels=['A Tiempo', 'Con Demora'])
        axes[i].set_title(f'Matriz de Confusión - {name.upper()}', fontweight='bold')
        axes[i].set_xlabel('Predicción')
        axes[i].set_ylabel('Real')
    
    plt.tight_layout()
    plt.savefig('data/results/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Matrices de confusión guardadas en: data/results/confusion_matrices_comparison.png")

def plot_roc_curves(models, X_test, y_test, probabilities):
    """
    Crear gráficos de curvas ROC.
    
    Args:
        models (dict): Diccionario con modelos
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        probabilities (dict): Diccionario con probabilidades
    """
    print("Generando curvas ROC...")
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if name in probabilities:
            y_proba = probabilities[name]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            
            plt.plot(fpr, tpr, label=f'{name.upper()} (AUC = {auc_score:.3f})', linewidth=2)
    
    # Línea de referencia (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio', alpha=0.5)
    
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/results/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Curvas ROC guardadas en: data/results/roc_curves_comparison.png")

def plot_metrics_comparison(comparison_df):
    """
    Crear gráfico de comparación de métricas.
    
    Args:
        comparison_df (pd.DataFrame): Tabla comparativa
    """
    print("Generando gráfico de comparación de métricas...")
    
    # Preparar datos para gráfico
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    available_metrics = [col for col in metrics_to_plot if col in comparison_df.columns]
    
    if not available_metrics:
        print("No hay métricas disponibles para graficar")
        return
    
    # Crear gráfico de barras
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(comparison_df.index))
    width = 0.15
    
    for i, metric in enumerate(available_metrics):
        ax.bar(x + i*width, comparison_df[metric], width, 
               label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_xlabel('Modelos', fontsize=12)
    ax.set_ylabel('Puntuación', fontsize=12)
    ax.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(available_metrics)-1) / 2)
    ax.set_xticklabels(comparison_df.index)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/results/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Gráfico de métricas guardado en: data/results/metrics_comparison.png")

def generate_detailed_reports(models, X_test, y_test, predictions):
    """
    Generar reportes detallados para cada modelo.
    
    Args:
        models (dict): Diccionario con modelos
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        predictions (dict): Diccionario con predicciones
    """
    print("Generando reportes detallados...")
    
    for name, model in models.items():
        y_pred = predictions[name]
        
        print(f"\n{'='*50}")
        print(f"REPORTE DETALLADO - {name.upper()}")
        print(f"{'='*50}")
        
        # Reporte de clasificación
        print("\nREPORTE DE CLASIFICACIÓN:")
        print("-" * 30)
        print(classification_report(y_test, y_pred, target_names=['A Tiempo', 'Con Demora']))
        
        # Matriz de confusión detallada
        cm = confusion_matrix(y_test, y_pred)
        print("\nMATRIZ DE CONFUSIÓN:")
        print("-" * 25)
        print(f"                 Predicción")
        print(f"                 A Tiempo  Con Demora")
        print(f"Real A Tiempo     {cm[0,0]:8d}  {cm[0,1]:10d}")
        print(f"Real Con Demora   {cm[1,0]:8d}  {cm[1,1]:10d}")
        
        # Calcular métricas adicionales
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nMÉTRICAS ADICIONALES:")
        print(f"Especificidad: {specificity:.4f}")
        print(f"Sensibilidad: {sensitivity:.4f}")

def save_results(comparison_df):
    """
    Guardar resultados de la evaluación.
    
    Args:
        comparison_df (pd.DataFrame): Tabla comparativa
    """
    print("Guardando resultados de evaluación...")
    
    # Crear directorio de resultados si no existe
    os.makedirs('data/results', exist_ok=True)
    
    # Guardar tabla comparativa
    comparison_df.to_csv('data/results/model_comparison.csv')
    print("Tabla comparativa guardada en: data/results/model_comparison.csv")
    
    # Crear reporte final
    report_content = f"""
REPORTE FINAL DE EVALUACIÓN DE MODELOS
=====================================

Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

RESUMEN DE RESULTADOS:
{comparison_df.to_string()}

MEJOR MODELO POR MÉTRICA:
"""
    
    for metric in comparison_df.columns:
        if metric != 'model':
            best_model = comparison_df[metric].idxmax()
            best_score = comparison_df.loc[best_model, metric]
            report_content += f"- {metric.upper()}: {best_model.upper()} ({best_score:.4f})\n"
    
    report_content += f"""
CONCLUSIONES:
- Se evaluaron {len(comparison_df)} modelos
- Métricas calculadas: {', '.join(comparison_df.columns)}
- Todos los modelos fueron evaluados en el mismo conjunto de prueba
- Los resultados están guardados en data/results/

ARCHIVOS GENERADOS:
- model_comparison.csv: Tabla comparativa de métricas
- confusion_matrices_comparison.png: Matrices de confusión
- roc_curves_comparison.png: Curvas ROC
- metrics_comparison.png: Gráfico de comparación de métricas
"""
    
    # Guardar reporte
    with open('data/results/evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("Reporte final guardado en: data/results/evaluation_report.txt")

def main():
    """
    Función principal para evaluar y comparar modelos.
    """
    print("="*60)
    print("EVALUACIÓN Y COMPARACIÓN DE MODELOS")
    print("="*60)
    
    # 1. Cargar modelos y datos
    models, X_test, y_test = load_models_and_data()
    
    if not models:
        print("Error: No se encontraron modelos para evaluar")
        return
    
    # 2. Crear tabla comparativa
    comparison_df, predictions, probabilities = create_comparison_table(models, X_test, y_test)
    
    # 3. Generar visualizaciones
    plot_confusion_matrices(models, X_test, y_test, predictions)
    plot_roc_curves(models, X_test, y_test, probabilities)
    plot_metrics_comparison(comparison_df)
    
    # 4. Generar reportes detallados
    generate_detailed_reports(models, X_test, y_test, predictions)
    
    # 5. Guardar resultados
    save_results(comparison_df)
    
    print("\n" + "="*60)
    print("EVALUACIÓN COMPLETADA EXITOSAMENTE")
    print("="*60)
    
    return comparison_df

if __name__ == "__main__":
    results = main()
