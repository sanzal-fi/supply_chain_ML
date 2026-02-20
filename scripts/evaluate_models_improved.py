"""
Script de Evaluación y Comparación de Modelos - Versión Mejorada
=================================================================

Este script evalúa y compara los modelos entrenados (KNN, SVM, XGBoost)
generando métricas, visualizaciones y reportes comparativos avanzados.

Mejoras:
- Rutas dinámicas (no hardcoded)
- Curvas Precision-Recall para datos desbalanceados
- Métricas de negocio (análisis costo-beneficio)
- Evaluación en train y test (detección de overfitting)
- Mejor manejo de errores
- Análisis de calibración de probabilidades

Autor: Sistema de Predicción de Demoras
Fecha: 2024
Versión: 2.0 (Mejorada)
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           roc_curve, classification_report, precision_recall_curve,
                           average_precision_score)
from sklearn.calibration import calibration_curve
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')
sns.set_palette("husl")

def get_project_paths():
    """
    Obtener rutas del proyecto de forma dinámica.
    
    Returns:
        dict: Diccionario con rutas del proyecto
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    paths = {
        'project_root': project_root,
        'data_processed': os.path.join(project_root, 'data', 'processed'),
        'models': os.path.join(project_root, 'models'),
        'results': os.path.join(project_root, 'data', 'results')
    }
    
    # Crear directorio de resultados si no existe
    os.makedirs(paths['results'], exist_ok=True)
    
    return paths

def load_models_and_data(paths):
    """
    Cargar modelos entrenados y datos de prueba y entrenamiento.
    
    Args:
        paths (dict): Diccionario con rutas del proyecto
        
    Returns:
        tuple: (models, X_train, X_test, y_train, y_test)
    """
    print("Cargando modelos y datos...")
    
    # Cargar datos de prueba
    try:
        X_test = pd.read_csv(os.path.join(paths['data_processed'], 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(paths['data_processed'], 'y_test.csv')).iloc[:, 0]
        print(f"✅ Datos de prueba cargados: {X_test.shape[0]:,} muestras, {X_test.shape[1]} características")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"❌ Error: No se encontraron los datos de prueba en {paths['data_processed']}")
    
    # Cargar datos de entrenamiento (para detectar overfitting)
    try:
        X_train = pd.read_csv(os.path.join(paths['data_processed'], 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(paths['data_processed'], 'y_train.csv')).iloc[:, 0]
        print(f"✅ Datos de entrenamiento cargados: {X_train.shape[0]:,} muestras")
    except FileNotFoundError:
        print("⚠️  Advertencia: No se encontraron datos de entrenamiento (no se detectará overfitting)")
        X_train, y_train = None, None
    
    # Cargar modelos
    models = {}
    model_names = ['knn', 'svm', 'xgboost']
    
    for name in model_names:
        model_path = os.path.join(paths['models'], f'{name}_model.pkl')
        try:
            models[name] = joblib.load(model_path)
            print(f"✅ Modelo {name.upper()} cargado exitosamente")
        except FileNotFoundError:
            print(f"⚠️  Advertencia: No se encontró el modelo {name.upper()} en {model_path}")
        except Exception as e:
            print(f"❌ Error al cargar {name.upper()}: {str(e)}")
    
    if not models:
        raise ValueError("❌ Error crítico: No se pudo cargar ningún modelo")
    
    print(f"\n📊 Modelos disponibles para evaluación: {', '.join([m.upper() for m in models.keys()])}")
    return models, X_train, X_test, y_train, y_test

def evaluate_single_model(model, model_name, X_test, y_test, X_train=None, y_train=None):
    """
    Evaluar un modelo individual en train y test.
    
    Args:
        model: Modelo entrenado
        model_name (str): Nombre del modelo
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        X_train: Características de entrenamiento (opcional)
        y_train: Variable objetivo de entrenamiento (opcional)
        
    Returns:
        dict: Métricas de evaluación
    """
    print(f"\n📈 Evaluando modelo {model_name.upper()}...")
    
    # Realizar predicciones en test
    y_pred_test = model.predict(X_test)
    
    # Calcular métricas en test
    metrics = {
        'model': model_name,
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
        'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
        'test_f1_score': f1_score(y_test, y_pred_test, zero_division=0),
    }
    
    # Calcular métricas en train (si disponible) para detectar overfitting
    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train)
        metrics['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        metrics['train_f1_score'] = f1_score(y_train, y_pred_train, zero_division=0)
        metrics['overfitting_gap'] = metrics['train_accuracy'] - metrics['test_accuracy']
    
    # Obtener probabilidades para métricas avanzadas (si el modelo las soporta)
    try:
        y_proba_test = model.predict_proba(X_test)[:, 1]
        metrics['test_roc_auc'] = roc_auc_score(y_test, y_proba_test)
        metrics['test_avg_precision'] = average_precision_score(y_test, y_proba_test)
        metrics['has_probabilities'] = True
    except (AttributeError, TypeError):
        y_proba_test = None
        metrics['test_roc_auc'] = np.nan
        metrics['test_avg_precision'] = np.nan
        metrics['has_probabilities'] = False
        print(f"  ⚠️  {model_name.upper()} no soporta predicción de probabilidades")
    
    # Calcular especificidad y sensibilidad
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    metrics['test_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['test_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics, y_pred_test, y_proba_test

def calculate_business_metrics(y_test, y_pred, cost_fp=100, cost_fn=500, benefit_tp=200):
    """
    Calcular métricas de negocio basadas en costos y beneficios.
    
    Args:
        y_test: Variable objetivo real
        y_pred: Predicciones del modelo
        cost_fp (float): Costo de un falso positivo (prepararse para demora que no ocurre)
        cost_fn (float): Costo de un falso negativo (no prepararse para demora real)
        benefit_tp (float): Beneficio de predecir correctamente una demora
        
    Returns:
        dict: Métricas de negocio
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calcular impacto financiero
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    total_benefit = tp * benefit_tp
    net_value = total_benefit - total_cost
    
    # Calcular métricas de negocio
    business_metrics = {
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'cost_false_positives': fp * cost_fp,
        'cost_false_negatives': fn * cost_fn,
        'benefit_true_positives': tp * benefit_tp,
        'total_cost': total_cost,
        'total_benefit': total_benefit,
        'net_business_value': net_value,
        'roi': (net_value / total_cost * 100) if total_cost > 0 else 0
    }
    
    return business_metrics

def create_comparison_table(models, X_test, y_test, X_train=None, y_train=None):
    """
    Crear tabla comparativa de métricas.
    
    Args:
        models (dict): Diccionario con modelos
        X_test: Características de prueba
        y_test: Variable objetivo de prueba
        X_train: Características de entrenamiento (opcional)
        y_train: Variable objetivo de entrenamiento (opcional)
        
    Returns:
        tuple: (comparison_df, predictions, probabilities, business_metrics_all)
    """
    print("\n" + "="*60)
    print("CREANDO TABLA COMPARATIVA DE MODELOS")
    print("="*60)
    
    results = []
    predictions = {}
    probabilities = {}
    business_metrics_all = {}
    
    for name, model in models.items():
        metrics, y_pred, y_proba = evaluate_single_model(
            model, name, X_test, y_test, X_train, y_train
        )
        results.append(metrics)
        predictions[name] = y_pred
        if y_proba is not None:
            probabilities[name] = y_proba
        
        # Calcular métricas de negocio
        business_metrics_all[name] = calculate_business_metrics(y_test, y_pred)
    
    # Crear DataFrame comparativo
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('model')
    
    # Redondear métricas numéricas
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].round(4)
    
    print("\n" + "="*60)
    print("TABLA COMPARATIVA DE MÉTRICAS TÉCNICAS")
    print("="*60)
    print(comparison_df.to_string())
    
    # Mostrar métricas de negocio
    print("\n" + "="*60)
    print("TABLA COMPARATIVA DE MÉTRICAS DE NEGOCIO")
    print("="*60)
    business_df = pd.DataFrame(business_metrics_all).T
    business_df = business_df.round(2)
    print(business_df[['net_business_value', 'roi', 'false_positives', 'false_negatives']].to_string())
    
    return comparison_df, predictions, probabilities, business_metrics_all

def plot_confusion_matrices(models, y_test, predictions, paths):
    """
    Crear gráficos de matrices de confusión.
    
    Args:
        models (dict): Diccionario con modelos
        y_test: Variable objetivo de prueba
        predictions (dict): Diccionario con predicciones
        paths (dict): Diccionario con rutas del proyecto
    """
    print("\n📊 Generando matrices de confusión...")
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = predictions[name]
        cm = confusion_matrix(y_test, y_pred)
        
        # Calcular porcentajes
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Crear anotaciones con conteos y porcentajes
        annot = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                          for j in range(cm.shape[1])] 
                         for i in range(cm.shape[0])])
        
        # Crear heatmap
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[i],
                   xticklabels=['A Tiempo', 'Con Demora'],
                   yticklabels=['A Tiempo', 'Con Demora'],
                   cbar_kws={'label': 'Conteo'})
        axes[i].set_title(f'Matriz de Confusión - {name.upper()}', fontweight='bold', fontsize=12)
        axes[i].set_xlabel('Predicción', fontsize=10)
        axes[i].set_ylabel('Real', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(paths['results'], 'confusion_matrices_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Matrices de confusión guardadas en: {save_path}")

def plot_roc_curves(models, y_test, probabilities, paths):
    """
    Crear gráficos de curvas ROC.
    
    Args:
        models (dict): Diccionario con modelos
        y_test: Variable objetivo de prueba
        probabilities (dict): Diccionario con probabilidades
        paths (dict): Diccionario con rutas del proyecto
    """
    if not probabilities:
        print("⚠️  No hay modelos con probabilidades para generar curvas ROC")
        return
    
    print("\n📊 Generando curvas ROC...")
    
    plt.figure(figsize=(10, 8))
    
    for name in models.keys():
        if name in probabilities:
            y_proba = probabilities[name]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            
            plt.plot(fpr, tpr, label=f'{name.upper()} (AUC = {auc_score:.3f})', linewidth=2.5)
    
    # Línea de referencia (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio (AUC = 0.500)', alpha=0.5, linewidth=1.5)
    
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12, fontweight='bold')
    plt.title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(paths['results'], 'roc_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Curvas ROC guardadas en: {save_path}")

def plot_precision_recall_curves(models, y_test, probabilities, paths):
    """
    Crear curvas Precision-Recall (mejor para datos desbalanceados).
    
    Args:
        models (dict): Diccionario con modelos
        y_test: Variable objetivo de prueba
        probabilities (dict): Diccionario con probabilidades
        paths (dict): Diccionario con rutas del proyecto
    """
    if not probabilities:
        print("⚠️  No hay modelos con probabilidades para generar curvas Precision-Recall")
        return
    
    print("\n📊 Generando curvas Precision-Recall...")
    
    plt.figure(figsize=(10, 8))
    
    # Calcular línea base (proporción de positivos)
    baseline = y_test.mean()
    
    for name in models.keys():
        if name in probabilities:
            y_proba = probabilities[name]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            avg_precision = average_precision_score(y_test, y_proba)
            
            plt.plot(recall, precision, 
                    label=f'{name.upper()} (AP = {avg_precision:.3f})', 
                    linewidth=2.5)
    
    # Línea base
    plt.axhline(y=baseline, color='k', linestyle='--', 
                label=f'Baseline (prevalencia = {baseline:.3f})', 
                alpha=0.5, linewidth=1.5)
    
    plt.xlabel('Recall (Sensibilidad)', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Curvas Precision-Recall - Comparación de Modelos\n(Mejor para Datos Desbalanceados)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    save_path = os.path.join(paths['results'], 'precision_recall_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Curvas Precision-Recall guardadas en: {save_path}")

def plot_calibration_curves(models, y_test, probabilities, paths):
    """
    Crear curvas de calibración de probabilidades.
    
    Args:
        models (dict): Diccionario con modelos
        y_test: Variable objetivo de prueba
        probabilities (dict): Diccionario con probabilidades
        paths (dict): Diccionario con rutas del proyecto
    """
    if not probabilities:
        print("⚠️  No hay modelos con probabilidades para generar curvas de calibración")
        return
    
    print("\n📊 Generando curvas de calibración...")
    
    plt.figure(figsize=(10, 8))
    
    for name in models.keys():
        if name in probabilities:
            y_proba = probabilities[name]
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_proba, n_bins=10, strategy='uniform'
            )
            
            plt.plot(mean_predicted_value, fraction_of_positives, 
                    marker='o', linewidth=2, label=f'{name.upper()}')
    
    # Línea de calibración perfecta
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente Calibrado', linewidth=1.5)
    
    plt.xlabel('Probabilidad Predicha', fontsize=12, fontweight='bold')
    plt.ylabel('Fracción de Positivos', fontsize=12, fontweight='bold')
    plt.title('Curvas de Calibración - Comparación de Modelos\n(Qué tan confiables son las probabilidades)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(paths['results'], 'calibration_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Curvas de calibración guardadas en: {save_path}")

def plot_metrics_comparison(comparison_df, paths):
    """
    Crear gráfico de comparación de métricas.
    
    Args:
        comparison_df (pd.DataFrame): Tabla comparativa
        paths (dict): Diccionario con rutas del proyecto
    """
    print("\n📊 Generando gráfico de comparación de métricas...")
    
    # Seleccionar métricas principales de test
    metrics_to_plot = ['test_accuracy', 'test_precision', 'test_recall', 
                      'test_f1_score', 'test_roc_auc']
    available_metrics = [col for col in metrics_to_plot if col in comparison_df.columns]
    
    if not available_metrics:
        print("⚠️  No hay métricas disponibles para graficar")
        return
    
    # Crear gráfico de barras
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(comparison_df.index))
    width = 0.15
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(available_metrics)))
    
    for i, metric in enumerate(available_metrics):
        metric_values = comparison_df[metric].fillna(0)  # Reemplazar NaN con 0 para visualización
        bars = ax.bar(x + i*width, metric_values, width, 
                     label=metric.replace('test_', '').replace('_', ' ').title(), 
                     alpha=0.8, color=colors[i])
        
        # Añadir valores encima de las barras
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(comparison_df[metric].iloc[j]):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax.set_ylabel('Puntuación', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(available_metrics)-1) / 2)
    ax.set_xticklabels([name.upper() for name in comparison_df.index])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    save_path = os.path.join(paths['results'], 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico de métricas guardado en: {save_path}")

def plot_business_metrics_comparison(business_metrics_all, paths):
    """
    Crear gráfico de comparación de métricas de negocio.
    
    Args:
        business_metrics_all (dict): Diccionario con métricas de negocio
        paths (dict): Diccionario con rutas del proyecto
    """
    print("\n📊 Generando gráfico de comparación de métricas de negocio...")
    
    # Crear DataFrame de métricas de negocio
    business_df = pd.DataFrame(business_metrics_all).T
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Valor neto del negocio
    models_list = list(business_metrics_all.keys())
    net_values = [business_metrics_all[m]['net_business_value'] for m in models_list]
    colors = ['green' if v > 0 else 'red' for v in net_values]
    
    axes[0].bar(range(len(models_list)), net_values, color=colors, alpha=0.7)
    axes[0].set_xlabel('Modelos', fontweight='bold')
    axes[0].set_ylabel('Valor Neto del Negocio ($)', fontweight='bold')
    axes[0].set_title('Valor Neto del Negocio por Modelo', fontweight='bold')
    axes[0].set_xticks(range(len(models_list)))
    axes[0].set_xticklabels([m.upper() for m in models_list])
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for i, v in enumerate(net_values):
        axes[0].text(i, v, f'${v:,.0f}', ha='center', 
                    va='bottom' if v > 0 else 'top', fontweight='bold')
    
    # Gráfico 2: Errores (FP vs FN)
    fp_values = [business_metrics_all[m]['false_positives'] for m in models_list]
    fn_values = [business_metrics_all[m]['false_negatives'] for m in models_list]
    
    x = np.arange(len(models_list))
    width = 0.35
    
    axes[1].bar(x - width/2, fp_values, width, label='Falsos Positivos', alpha=0.7, color='orange')
    axes[1].bar(x + width/2, fn_values, width, label='Falsos Negativos', alpha=0.7, color='red')
    axes[1].set_xlabel('Modelos', fontweight='bold')
    axes[1].set_ylabel('Cantidad de Errores', fontweight='bold')
    axes[1].set_title('Comparación de Tipos de Error', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.upper() for m in models_list])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(paths['results'], 'business_metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico de métricas de negocio guardado en: {save_path}")

def generate_detailed_reports(models, y_test, predictions, paths):
    """
    Generar reportes detallados para cada modelo.
    
    Args:
        models (dict): Diccionario con modelos
        y_test: Variable objetivo de prueba
        predictions (dict): Diccionario con predicciones
        paths (dict): Diccionario con rutas del proyecto
    """
    print("\n📋 Generando reportes detallados por modelo...")
    
    for name, model in models.items():
        y_pred = predictions[name]
        
        report_content = f"""
{'='*60}
REPORTE DETALLADO - {name.upper()}
{'='*60}

REPORTE DE CLASIFICACIÓN:
{'-'*60}
{classification_report(y_test, y_pred, target_names=['A Tiempo', 'Con Demora'])}

MATRIZ DE CONFUSIÓN:
{'-'*60}
"""
        
        # Matriz de confusión detallada
        cm = confusion_matrix(y_test, y_pred)
        report_content += f"""
                 Predicción
                 A Tiempo  Con Demora
Real A Tiempo     {cm[0,0]:8d}  {cm[0,1]:10d}
Real Con Demora   {cm[1,0]:8d}  {cm[1,1]:10d}

"""
        
        # Calcular métricas adicionales
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        report_content += f"""
MÉTRICAS ADICIONALES:
{'-'*60}
Especificidad (True Negative Rate): {specificity:.4f}
Sensibilidad (True Positive Rate):  {sensitivity:.4f}
Verdaderos Negativos (TN):          {tn}
Falsos Positivos (FP):              {fp}
Falsos Negativos (FN):              {fn}
Verdaderos Positivos (TP):          {tp}
"""
        
        # Guardar reporte individual
        report_path = os.path.join(paths['results'], f'detailed_report_{name}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"  ✅ Reporte de {name.upper()} guardado en: {report_path}")

def save_results(comparison_df, business_metrics_all, paths):
    """
    Guardar resultados de la evaluación.
    
    Args:
        comparison_df (pd.DataFrame): Tabla comparativa de métricas técnicas
        business_metrics_all (dict): Métricas de negocio
        paths (dict): Diccionario con rutas del proyecto
    """
    print("\n💾 Guardando resultados de evaluación...")
    
    # Guardar tabla comparativa de métricas técnicas
    comparison_path = os.path.join(paths['results'], 'model_comparison.csv')
    comparison_df.to_csv(comparison_path)
    print(f"✅ Tabla comparativa técnica guardada en: {comparison_path}")
    
    # Guardar métricas de negocio
    business_df = pd.DataFrame(business_metrics_all).T
    business_path = os.path.join(paths['results'], 'business_metrics_comparison.csv')
    business_df.to_csv(business_path)
    print(f"✅ Tabla comparativa de negocio guardada en: {business_path}")
    
    # Crear reporte final consolidado
    report_content = f"""
{'='*70}
REPORTE FINAL DE EVALUACIÓN DE MODELOS
{'='*70}

Fecha y Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
RESUMEN DE MÉTRICAS TÉCNICAS:
{'='*70}
{comparison_df.to_string()}

{'='*70}
RESUMEN DE MÉTRICAS DE NEGOCIO:
{'='*70}
{business_df.to_string()}

{'='*70}
MEJOR MODELO POR MÉTRICA TÉCNICA:
{'='*70}
"""
    
    # Identificar mejor modelo por cada métrica
    for metric in comparison_df.columns:
        if metric not in ['has_probabilities', 'overfitting_gap']:
            valid_data = comparison_df[metric].dropna()
            if len(valid_data) > 0:
                if 'overfitting' in metric:
                    # Para overfitting, menor es mejor
                    best_model = valid_data.abs().idxmin()
                    best_score = valid_data.loc[best_model]
                else:
                    best_model = valid_data.idxmax()
                    best_score = valid_data.loc[best_model]
                report_content += f"- {metric.upper()}: {best_model.upper()} ({best_score:.4f})\n"
    
    report_content += f"""
{'='*70}
MEJOR MODELO POR MÉTRICA DE NEGOCIO:
{'='*70}
"""
    
    # Mejor modelo por valor neto de negocio
    best_business_model = max(business_metrics_all.keys(), 
                             key=lambda x: business_metrics_all[x]['net_business_value'])
    best_net_value = business_metrics_all[best_business_model]['net_business_value']
    best_roi = business_metrics_all[best_business_model]['roi']
    
    report_content += f"""
- VALOR NETO DEL NEGOCIO: {best_business_model.upper()} (${best_net_value:,.2f})
- ROI (Return on Investment): {best_business_model.upper()} ({best_roi:.2f}%)

{'='*70}
RECOMENDACIÓN FINAL:
{'='*70}

Basado en el análisis técnico y de negocio, el modelo recomendado es:
>>> {best_business_model.upper()} <<<

Razones:
- Mejor valor neto del negocio: ${best_net_value:,.2f}
- ROI: {best_roi:.2f}%
- Balance óptimo entre precisión técnica e impacto comercial

{'='*70}
CONCLUSIONES:
{'='*70}
- Se evaluaron {len(comparison_df)} modelos
- Métricas técnicas calculadas: {', '.join([c for c in comparison_df.columns if c not in ['has_probabilities']])}
- Métricas de negocio calculadas: Valor neto, ROI, costos por tipo de error
- Todos los modelos fueron evaluados en el mismo conjunto de prueba
- Se generaron curvas ROC, Precision-Recall y de calibración
- Los resultados consideran tanto precisión técnica como impacto comercial

{'='*70}
ARCHIVOS GENERADOS:
{'='*70}
- model_comparison.csv: Tabla comparativa de métricas técnicas
- business_metrics_comparison.csv: Tabla comparativa de métricas de negocio
- confusion_matrices_comparison.png: Matrices de confusión
- roc_curves_comparison.png: Curvas ROC
- precision_recall_curves.png: Curvas Precision-Recall (mejor para desbalance)
- calibration_curves.png: Curvas de calibración de probabilidades
- metrics_comparison.png: Gráfico de comparación de métricas técnicas
- business_metrics_comparison.png: Gráfico de métricas de negocio
- detailed_report_[modelo].txt: Reportes detallados por modelo

{'='*70}
PRÓXIMOS PASOS RECOMENDADOS:
{'='*70}
1. Revisar las curvas de calibración para ajustar umbrales de decisión
2. Analizar el trade-off entre falsos positivos y falsos negativos
3. Considerar ajustar costos de negocio según la realidad operativa
4. Implementar el modelo recomendado en producción con monitoreo continuo
5. Realizar pruebas A/B si es posible antes del despliegue completo

{'='*70}
"""
    
    # Guardar reporte final
    report_path = os.path.join(paths['results'], 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Reporte final consolidado guardado en: {report_path}")

def main():
    """
    Función principal para evaluar y comparar modelos.
    """
    print("="*60)
    print("EVALUACIÓN Y COMPARACIÓN DE MODELOS - VERSIÓN MEJORADA")
    print("="*60)
    
    try:
        # 1. Obtener rutas del proyecto
        paths = get_project_paths()
        
        # 2. Cargar modelos y datos
        models, X_train, X_test, y_train, y_test = load_models_and_data(paths)
        
        # 3. Crear tabla comparativa
        comparison_df, predictions, probabilities, business_metrics_all = create_comparison_table(
            models, X_test, y_test, X_train, y_train
        )
        
        # 4. Generar visualizaciones
        plot_confusion_matrices(models, y_test, predictions, paths)
        plot_roc_curves(models, y_test, probabilities, paths)
        plot_precision_recall_curves(models, y_test, probabilities, paths)
        plot_calibration_curves(models, y_test, probabilities, paths)
        plot_metrics_comparison(comparison_df, paths)
        plot_business_metrics_comparison(business_metrics_all, paths)
        
        # 5. Generar reportes detallados
        generate_detailed_reports(models, y_test, predictions, paths)
        
        # 6. Guardar resultados
        save_results(comparison_df, business_metrics_all, paths)
        
        print("\n" + "="*60)
        print("✅ EVALUACIÓN COMPLETADA EXITOSAMENTE")
        print("="*60)
        print(f"\n📁 Todos los resultados guardados en: {paths['results']}")
        
        # Mostrar recomendación final
        best_model = max(business_metrics_all.keys(), 
                        key=lambda x: business_metrics_all[x]['net_business_value'])
        print(f"\n🏆 MODELO RECOMENDADO: {best_model.upper()}")
        print(f"   Valor neto del negocio: ${business_metrics_all[best_model]['net_business_value']:,.2f}")
        print(f"   ROI: {business_metrics_all[best_model]['roi']:.2f}%")
        
        return comparison_df, business_metrics_all
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR EN LA EVALUACIÓN")
        print("="*60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    results, business_metrics = main()