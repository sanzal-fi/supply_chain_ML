# Modelo de Predicción de Demoras en Entregas

Este proyecto desarrolla un modelo de aprendizaje supervisado para predecir si una orden de compra será entregada a tiempo o con demora, utilizando algoritmos de clasificación y análisis de características comerciales, logísticas y geográficas.

## 🎯 Objetivo

Construir una herramienta que permita anticiparse con precisión a las demoras logísticas, de forma tal que la empresa pueda actuar proactivamente para mejorar su desempeño operativo y su nivel de servicio.

## 📊 Dataset

- **Archivo**: `DataCoSupplyChainDataset.csv`
- **Registros**: 180,519 órdenes
- **Variables**: 53 características incluyendo información de clientes, productos, órdenes y envíos
- **Variable Objetivo**: `demora` (1 si hay demora, 0 si es a tiempo)

## 🏗️ Estructura del Proyecto

```
trabajo_practico/
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb    # Análisis exploratorio completo
├── scripts/
│   ├── data_preprocessing.py              # Preprocesamiento de datos
│   ├── train_knn.py                       # Entrenamiento KNN
│   ├── train_svm.py                       # Entrenamiento SVM
│   ├── train_xgboost.py                   # Entrenamiento XGBoost
│   └── evaluate_models.py                 # Evaluación y comparación
├── data/
│   ├── processed/                         # Datos preprocesados
│   └── results/                           # Resultados y métricas
├── models/                                # Modelos entrenados
└── requirements.txt                       # Dependencias
```

## 🚀 Instalación

1. **Clonar o descargar el proyecto**
2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Dependencias

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- jupyter
- joblib

## 🔄 Flujo de Trabajo

### 1. Análisis Exploratorio de Datos (EDA)
**Archivo**: `notebooks/01_analisis_exploratorio.ipynb`

- Carga y exploración inicial del dataset
- Creación de variable objetivo `demora`
- Análisis de calidad de datos (nulos, duplicados)
- Detección y tratamiento de outliers
- Análisis univariado y bivariado
- Identificación de variables relevantes

### 2. Preprocesamiento de Datos
**Archivo**: `scripts/data_preprocessing.py`

```bash
python scripts/data_preprocessing.py
```

**Funciones principales**:
- Eliminación de columnas irrelevantes/sensibles
- Tratamiento de valores nulos y duplicados
- Codificación one-hot de variables categóricas
- Estandarización de variables numéricas
- División estratificada train/test (80/20)

### 3. Entrenamiento de Modelos

#### KNN (K-Nearest Neighbors)
```bash
python scripts/train_knn.py
```

#### SVM (Support Vector Machine)
```bash
python scripts/train_svm.py
```

#### XGBoost (Extreme Gradient Boosting)
```bash
python scripts/train_xgboost.py
```

**Características de entrenamiento**:
- Búsqueda de hiperparámetros con GridSearchCV
- Validación cruzada estratificada (5-fold)
- Optimización basada en F1-Score
- Guardado automático de modelos y parámetros

### 4. Evaluación y Comparación
**Archivo**: `scripts/evaluate_models.py`

```bash
python scripts/evaluate_models.py
```

**Métricas evaluadas**:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

**Visualizaciones generadas**:
- Matrices de confusión
- Curvas ROC
- Comparación de métricas
- Reportes detallados

## 📈 Resultados

Los resultados se guardan automáticamente en `data/results/`:

- `model_comparison.csv`: Tabla comparativa de métricas
- `confusion_matrices_comparison.png`: Matrices de confusión
- `roc_curves_comparison.png`: Curvas ROC
- `metrics_comparison.png`: Gráfico de comparación
- `evaluation_report.txt`: Reporte final
- `xgboost_feature_importance.csv`: Importancia de características (XGBoost)

## 🎯 Algoritmos Utilizados

### 1. K-Nearest Neighbors (KNN)
- **Hiperparámetros optimizados**: n_neighbors, weights, metric
- **Ventajas**: Simple, no paramétrico, bueno para datos no lineales
- **Desventajas**: Computacionalmente costoso, sensible a outliers

### 2. Support Vector Machine (SVM)
- **Hiperparámetros optimizados**: C, kernel, gamma
- **Ventajas**: Efectivo en espacios de alta dimensión, robusto
- **Desventajas**: Lento con datasets grandes, sensible a escalado

### 3. XGBoost (Extreme Gradient Boosting)
- **Hiperparámetros optimizados**: n_estimators, max_depth, learning_rate, subsample
- **Ventajas**: Alta precisión, manejo de missing values, importancia de características
- **Desventajas**: Puede sobreajustar, más complejo

## 🔧 Uso de los Modelos

### Cargar un modelo entrenado:
```python
import joblib

# Cargar modelo
model = joblib.load('models/knn_model.pkl')

# Cargar scaler
scaler = joblib.load('data/processed/scaler.pkl')

# Hacer predicción
prediction = model.predict(new_data)
```

### Preprocesar nuevos datos:
```python
from scripts.data_preprocessing import clean_data, encode_categorical, scale_features

# Aplicar mismo preprocesamiento que en entrenamiento
processed_data = clean_data(new_data)
processed_data = encode_categorical(processed_data)
processed_data, _ = scale_features(processed_data)
```

## 📊 Interpretación de Resultados

### Métricas Clave:
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Proporción de predicciones positivas que son correctas
- **Recall**: Proporción de casos positivos detectados correctamente
- **F1-Score**: Media armónica entre precision y recall
- **AUC-ROC**: Área bajo la curva ROC (capacidad de discriminación)

### Matriz de Confusión:
```
                 Predicción
                 A Tiempo  Con Demora
Real A Tiempo     TN        FP
Real Con Demora   FN        TP
```

## 🎯 Factores Clave Identificados

Basado en el análisis exploratorio, los factores más relevantes para predecir demoras incluyen:

- **Variables de tiempo**: Días de envío programados vs reales
- **Modo de envío**: Tipo de transporte utilizado
- **Región geográfica**: Ubicación de destino
- **Volumen del pedido**: Cantidad y valor de productos
- **Segmento de cliente**: Tipo de cliente
- **Categoría de producto**: Tipo de productos

## 🔄 Reproducibilidad

- **Semilla aleatoria**: `random_state=42` en todas las operaciones
- **División estratificada**: Mantiene proporción de clases
- **Validación cruzada**: 5-fold estratificada
- **Escalado consistente**: Mismo scaler para entrenamiento y prueba

## 📝 Notas Técnicas

- **Balance de clases**: Se analiza la distribución de la variable objetivo
- **Tratamiento de outliers**: Método IQR con capping
- **Codificación categórica**: One-hot encoding para todas las variables categóricas
- **Escalado**: StandardScaler para variables numéricas
- **Validación**: Separación estricta entre entrenamiento y prueba

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte de un trabajo práctico de Ciencia de Datos, implementando las mejores prácticas en machine learning y análisis de datos.

## 📄 Licencia

Este proyecto es de uso académico y educativo.

---

**Desarrollado con ❤️ para la comunidad de Ciencia de Datos**
# supply_chain_tp
