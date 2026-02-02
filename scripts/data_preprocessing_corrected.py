"""
Script de Preprocesamiento de Datos - Versión Optimizada
=========================================================

Este script contiene todas las funciones necesarias para preprocesar
el dataset DataCoSupplyChainDataset.csv con técnicas avanzadas de
reducción de dimensionalidad y selección de características.

Autor: Sistema de Predicción de Demoras
Fecha: 2024
Versión: 2.0 (Optimizada para Alto Rendimiento)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Cargar el dataset original desde el archivo CSV.
    
    Args:
        file_path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    print("Cargando dataset...")
    
    # Lista de codificaciones comunes a probar
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    # Delimitadores comunes a probar
    delimiters = [';', ',', '\t']
    
    df = None
    successful_encoding = None
    successful_delimiter = None
    
    # Intentar diferentes combinaciones de encoding y delimiter
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                print(f"Probando codificación: {encoding} con delimiter: '{delimiter}'")
                df_temp = pd.read_csv(file_path, encoding=encoding, sep=delimiter)
                
                # Verificar que se leyeron múltiples columnas
                if df_temp.shape[1] > 1:
                    df = df_temp
                    successful_encoding = encoding
                    successful_delimiter = delimiter
                    print(f"✅ Éxito con codificación: {encoding} y delimiter: '{delimiter}'")
                    break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                continue
        
        if df is not None:
            break
    
    if df is None or df.shape[1] <= 1:
        print("❌ No se pudo cargar el archivo con ninguna codificación/delimiter estándar")
        print("Intentando con encoding='latin-1', delimiter=';' y manejo de errores...")
        df = pd.read_csv(file_path, encoding='latin-1', sep=';', encoding_errors='replace')
    
    print(f"Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    
    # Limpiar nombres de columnas (eliminar espacios y caracteres extraños)
    df.columns = df.columns.str.strip()
    
    return df

def create_target_variable(df):
    """
    Crear la variable objetivo 'demora' comparando días reales vs programados.
    
    Args:
        df (pd.DataFrame): Dataset original
        
    Returns:
        pd.DataFrame: Dataset con variable objetivo agregada
    """
    print("Creando variable objetivo 'demora'...")
    
    # Validar que existen las columnas necesarias
    required_cols = ['Days for shipping (real)', 'Days for shipment (scheduled)']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Error: Faltan columnas requeridas: {missing}")
    
    df['demora'] = (df['Days for shipping (real)'] > df['Days for shipment (scheduled)']).astype(int)
    
    # Validar que la variable objetivo tiene ambas clases
    unique_values = df['demora'].unique()
    if len(unique_values) < 2:
        raise ValueError(f"❌ Error: La variable objetivo solo tiene una clase: {unique_values}")
    
    # Mostrar distribución
    demora_0 = sum(df['demora'] == 0)
    demora_1 = sum(df['demora'] == 1)
    total = len(df)
    
    print(f"Entregas a tiempo (demora=0): {demora_0:,} ({demora_0/total*100:.1f}%)")
    print(f"Entregas con demora (demora=1): {demora_1:,} ({demora_1/total*100:.1f}%)")
    
    return df

def extract_date_features(df):
    """
    Extraer características de la fecha de orden.
    
    Args:
        df (pd.DataFrame): Dataset original
        
    Returns:
        pd.DataFrame: Dataset con características de fecha extraídas
    """
    print("Extrayendo características de fecha...")
    
    # Convertir columna de fecha a datetime
    date_col = 'order date (DateOrders)'
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Extraer características temporales
        df['order_year'] = df[date_col].dt.year
        df['order_month'] = df[date_col].dt.month
        df['order_day'] = df[date_col].dt.day
        df['order_weekday'] = df[date_col].dt.dayofweek
        df['order_quarter'] = df[date_col].dt.quarter
        
        print(f"Extraídas 5 características temporales de '{date_col}'")
        
        # Eliminar la columna de fecha original
        df = df.drop(columns=[date_col])
    else:
        print(f"⚠️ Columna '{date_col}' no encontrada")
    
    return df

def clean_data(df):
    """
    Limpiar el dataset eliminando columnas irrelevantes, duplicados y nulos.
    
    Args:
        df (pd.DataFrame): Dataset con variable objetivo
        
    Returns:
        pd.DataFrame: Dataset limpio
    """
    print("Limpiando datos...")
    
    # Columnas a eliminar (irrelevantes, sensibles o con fuga de información)
    columns_to_drop = [
        # === FUGA DE INFORMACIÓN (conocidas solo después del resultado) ===
        'Days for shipping (real)',       # Usado para crear target, contiene la respuesta
        'Late_delivery_risk',              # Derivado del resultado real
        'Order Status',                    # Conocido después de la entrega
        'Delivery Status',                 # Conocido después de la entrega
        'shipping date (DateOrders)',      # Fecha real de envío (posterior al pedido)
        
        # === DATOS SENSIBLES/PII ===
        'Customer Password',
        'Customer Email',
        'Customer Fname',
        'Customer Lname',
        'Customer Street',
        
        # === IDENTIFICADORES DE ALTA CARDINALIDAD (no predictivos) ===
        'Order Id',                        # Identificador único
        'Order Item Id',                   # Identificador único
        'Order Item Cardprod Id',          # Identificador único
        'Product Card Id',                 # Identificador único
        'Product Category Id',             # Redundante con Category Name
        'Category Id',                     # Redundante con Category Name
        'Department Id',                   # Redundante con Department Name
        'Customer Id',                     # Identificador único
        'Order Customer Id',               # Redundante con Customer Id
        
        # === CONTENIDO DE ALTA DIMENSIONALIDAD SIN VALOR PREDICTIVO ===
        'Product Description',             # Texto libre
        'Product Image',                   # URL/ruta de imagen
        
        # === COLUMNA INICIAL SIN NOMBRE (índice) ===
        ';',                               # Primera columna sin nombre
    ]
    
    # Filtrar solo las que existen
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    print(f"Eliminando {len(existing_columns_to_drop)} columnas irrelevantes/sensibles...")
    
    df_clean = df.drop(columns=existing_columns_to_drop, errors='ignore')
    
    # Eliminar duplicados
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    print(f"Eliminados {duplicates_removed:,} registros duplicados")
    
    # Manejar valores nulos
    null_counts = df_clean.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    
    if len(columns_with_nulls) > 0:
        print(f"Columnas con valores nulos: {len(columns_with_nulls)}")
        for col, count in columns_with_nulls.items():
            print(f"  {col}: {count:,} ({count/len(df_clean)*100:.2f}%)")
        
        # Para variables numéricas, llenar con la mediana
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                print(f"  Llenado {col} con mediana")
        
        # Para variables categóricas, llenar con la moda
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_value = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_value, inplace=True)
                print(f"  Llenado {col} con moda: {mode_value}")
    else:
        print("No se encontraron valores nulos")
    
    print(f"Dataset limpio: {df_clean.shape[0]:,} filas, {df_clean.shape[1]} columnas")
    return df_clean

def reduce_geographic_redundancy(df):
    """
    Reducir redundancia geográfica manteniendo solo los niveles más informativos.
    Estrategia: Mantener las características más agregadas (región/país) y eliminar las muy específicas (ciudad/código postal).
    
    Args:
        df (pd.DataFrame): Dataset limpio
        
    Returns:
        pd.DataFrame: Dataset con redundancia geográfica reducida
    """
    print("Reduciendo redundancia geográfica...")
    
    # Columnas geográficas a eliminar (mantener solo las más informativas)
    geo_columns_to_drop = [
        # Customer: Eliminar City y Zipcode, mantener State y Country
        'Customer City',
        'Customer Zipcode',
        
        # Order: Eliminar City, State y Zipcode, mantener Region y Country
        'Order City',
        'Order State',
        'Order Zipcode',
    ]
    
    # Filtrar solo las que existen
    existing_geo_to_drop = [col for col in geo_columns_to_drop if col in df.columns]
    
    if existing_geo_to_drop:
        df = df.drop(columns=existing_geo_to_drop, errors='ignore')
        print(f"Eliminadas {len(existing_geo_to_drop)} columnas geográficas redundantes")
        print(f"Columnas eliminadas: {', '.join(existing_geo_to_drop)}")
    
    # Columnas geográficas mantenidas
    kept_geo = [col for col in ['Customer State', 'Customer Country', 'Order Region', 'Order Country', 'Market', 'Latitude', 'Longitude'] if col in df.columns]
    print(f"Columnas geográficas mantenidas: {', '.join(kept_geo)}")
    
    return df

def reduce_cardinality(df, max_categories=15):
    """
    Reducir la cardinalidad de variables categóricas agrupando categorías raras en 'Other'.
    Versión optimizada con límite agresivo para minimizar dimensionalidad.

    Args:
        df (pd.DataFrame): Dataset limpio
        max_categories (int): Número máximo de categorías a conservar por columna

    Returns:
        pd.DataFrame: Dataset con cardinalidad reducida
    """
    print(f"Reduciendo cardinalidad de variables categóricas (max={max_categories})...")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Excluir la variable objetivo
    if 'demora' in categorical_cols:
        categorical_cols.remove('demora')
    
    if not categorical_cols:
        print("No se encontraron variables categóricas para reducir")
        return df
    
    reduced_count = 0
    for col in categorical_cols:
        value_counts = df[col].value_counts(dropna=False)
        original_unique = len(value_counts)
        
        if original_unique > max_categories:
            # Mantener solo las top categorías
            top_categories = value_counts.index[:max_categories]
            df[col] = df[col].where(df[col].isin(top_categories), other='Other')
            reduced_unique = df[col].nunique(dropna=False)
            print(f"  {col}: {original_unique} -> {reduced_unique} categorías")
            reduced_count += 1
    
    print(f"Total de columnas con cardinalidad reducida: {reduced_count}")
    return df

def handle_outliers(df):
    """
    Tratar outliers en variables numéricas usando el método IQR (capping).
    
    Args:
        df (pd.DataFrame): Dataset limpio
        
    Returns:
        pd.DataFrame: Dataset con outliers tratados
    """
    print("Tratando outliers...")
    
    # Identificar dinámicamente todas las variables numéricas (excluyendo la variable objetivo)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Excluir la variable objetivo
    if 'demora' in numeric_cols:
        numeric_cols.remove('demora')
    
    if not numeric_cols:
        print("No se encontraron variables numéricas para tratar outliers")
        return df
    
    print(f"Variables numéricas a analizar: {len(numeric_cols)}")
    outliers_treated = 0
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Contar outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        
        if outlier_count > 0:
            # Cap outliers en lugar de eliminarlos (winsorización)
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            outliers_treated += outlier_count
            print(f"  {col}: {outlier_count:,} outliers capados")
    
    print(f"Total de outliers tratados: {outliers_treated:,}")
    return df

def encode_categorical(df):
    """
    Codificar variables categóricas usando one-hot encoding.
    
    Args:
        df (pd.DataFrame): Dataset con outliers tratados
        
    Returns:
        pd.DataFrame: Dataset con variables categóricas codificadas
    """
    print("Codificando variables categóricas...")
    
    # Identificar variables categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Excluir la variable objetivo si está presente
    if 'demora' in categorical_cols:
        categorical_cols.remove('demora')
    
    if not categorical_cols:
        print("No se encontraron variables categóricas para codificar")
        return df
    
    print(f"Variables categóricas a codificar: {len(categorical_cols)}")
    for col in categorical_cols:
        print(f"  {col}: {df[col].nunique()} categorías")
    
    # Aplicar one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
    
    print(f"Variables después del encoding: {df_encoded.shape[1]}")
    print(f"Variables agregadas por one-hot: {df_encoded.shape[1] - df.shape[1]}")
    
    return df_encoded

def remove_low_variance_features(X_train, X_test, threshold=0.01):
    """
    Eliminar características con baja varianza (casi constantes).
    
    Args:
        X_train (pd.DataFrame): Conjunto de entrenamiento
        X_test (pd.DataFrame): Conjunto de prueba
        threshold (float): Umbral de varianza
        
    Returns:
        tuple: (X_train_reduced, X_test_reduced, selected_features)
    """
    print(f"Eliminando características con varianza < {threshold}...")
    
    initial_features = X_train.shape[1]
    
    # Aplicar VarianceThreshold solo en datos de entrenamiento
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)
    
    # Obtener máscara de características seleccionadas
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()
    
    # Aplicar a ambos conjuntos
    X_train_reduced = X_train[selected_features]
    X_test_reduced = X_test[selected_features]
    
    removed = initial_features - len(selected_features)
    print(f"Eliminadas {removed} características con baja varianza")
    print(f"Características restantes: {len(selected_features)}")
    
    return X_train_reduced, X_test_reduced, selected_features

def remove_correlated_features(X_train, X_test, threshold=0.95):
    """
    Eliminar características altamente correlacionadas para reducir redundancia.
    
    Args:
        X_train (pd.DataFrame): Conjunto de entrenamiento
        X_test (pd.DataFrame): Conjunto de prueba
        threshold (float): Umbral de correlación para eliminar
        
    Returns:
        tuple: (X_train_reduced, X_test_reduced, selected_features)
    """
    print(f"Eliminando características correlacionadas (umbral={threshold})...")
    
    initial_features = X_train.shape[1]
    
    # Calcular matriz de correlación solo en entrenamiento
    print("Calculando matriz de correlación...")
    corr_matrix = X_train.corr().abs()
    
    # Seleccionar la parte superior del triángulo de la matriz de correlación
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Encontrar columnas con correlación > threshold
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    print(f"Encontradas {len(to_drop)} características altamente correlacionadas")
    
    # Eliminar características correlacionadas
    X_train_reduced = X_train.drop(columns=to_drop)
    X_test_reduced = X_test.drop(columns=to_drop)
    
    selected_features = X_train_reduced.columns.tolist()
    
    removed = initial_features - len(selected_features)
    print(f"Eliminadas {removed} características correlacionadas")
    print(f"Características restantes: {len(selected_features)}")
    
    return X_train_reduced, X_test_reduced, selected_features

def scale_train_test(X_train, X_test):
    """
    Estandarizar variables numéricas AJUSTANDO SOLO EN TRAIN y aplicando en TEST.

    Args:
        X_train (pd.DataFrame): Características de entrenamiento
        X_test (pd.DataFrame): Características de prueba

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("Estandarizando variables numéricas (fit en TRAIN, transform en TEST)...")

    # Identificar columnas numéricas
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("No se encontraron variables numéricas para escalar")
        return X_train, X_test, None
    
    print(f"Variables numéricas a escalar: {len(numeric_cols)}")

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("Escalado aplicado correctamente a TRAIN y TEST")
    return X_train_scaled, X_test_scaled, scaler

def split_data(df, target_col='demora', test_size=0.2, random_state=42):
    """
    Dividir el dataset en conjuntos de entrenamiento y prueba.
    
    Args:
        df (pd.DataFrame): Dataset preprocesado
        target_col (str): Nombre de la columna objetivo
        test_size (float): Proporción del conjunto de prueba
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Dividiendo datos en entrenamiento y prueba...")
    
    # Validar que existe la variable objetivo
    if target_col not in df.columns:
        raise ValueError(f"❌ Error: La variable objetivo '{target_col}' no existe en el dataset")
    
    # Separar características y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]:,} muestras, {X_train.shape[1]} características")
    print(f"Conjunto de prueba: {X_test.shape[0]:,} muestras, {X_test.shape[1]} características")
    
    # Verificar distribución de clases
    print(f"Distribución de clases en entrenamiento: {y_train.value_counts().to_dict()}")
    print(f"Distribución de clases en prueba: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, scaler, output_dir='data/processed'):
    """
    Guardar los datos procesados y objetos necesarios.
    
    Args:
        X_train, X_test, y_train, y_test: Conjuntos de datos
        scaler: Objeto scaler entrenado
        output_dir (str): Directorio de salida
    """
    print("Guardando datos procesados...")
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar conjuntos de datos
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    # Guardar scaler (solo si existe)
    if scaler is not None:
        joblib.dump(scaler, f'{output_dir}/scaler.pkl')
        print("  - scaler.pkl guardado")
    
    # Guardar nombres de características
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, f'{output_dir}/feature_names.pkl')
    
    print(f"Datos guardados en: {output_dir}")
    print("Archivos guardados:")
    print("  - X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("  - feature_names.pkl")

def main(
    correlation_threshold=0.90,
    variance_threshold=0.01,
    max_categories=15
):
    """
    Función principal que ejecuta todo el pipeline de preprocesamiento optimizado.
    
    Args:
        correlation_threshold (float): Umbral para eliminar características correlacionadas (0.90 = más agresivo)
        variance_threshold (float): Umbral para eliminar características de baja varianza
        max_categories (int): Número máximo de categorías por variable categórica
    """
    print("="*60)
    print("PIPELINE DE PREPROCESAMIENTO OPTIMIZADO")
    print("="*60)
    print(f"Configuración:")
    print(f"  - Umbral de correlación: {correlation_threshold}")
    print(f"  - Umbral de varianza: {variance_threshold}")
    print(f"  - Máximo de categorías: {max_categories}")
    print("="*60)
    
    try:
        # 1. Cargar datos
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        csv_path = os.path.join(project_root, 'DataCoSupplyChainDataset.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Error: No se encontró el archivo: {csv_path}")
        
        df = load_data(csv_path)
        
        # 2. Crear variable objetivo
        df = create_target_variable(df)
        
        # 3. Extraer características de fecha
        df = extract_date_features(df)
        
        # 4. Limpiar datos
        df = clean_data(df)
        
        # 5. Reducir redundancia geográfica
        df = reduce_geographic_redundancy(df)
        
        # 6. Tratar outliers
        df = handle_outliers(df)
        
        # 7. Reducir cardinalidad de categóricas (más agresivo)
        df = reduce_cardinality(df, max_categories=max_categories)

        # 8. Codificar variables categóricas
        df = encode_categorical(df)

        # 9. Dividir datos (antes de escalar y seleccionar características)
        X_train, X_test, y_train, y_test = split_data(df)
        
        print("\n" + "="*60)
        print("REDUCCIÓN DE DIMENSIONALIDAD")
        print("="*60)
        
        # 10. Eliminar características de baja varianza
        X_train, X_test, _ = remove_low_variance_features(
            X_train, X_test, threshold=variance_threshold
        )
        
        # 11. Eliminar características altamente correlacionadas
        X_train, X_test, _ = remove_correlated_features(
            X_train, X_test, threshold=correlation_threshold
        )
        
        print("\n" + "="*60)
        print("NORMALIZACIÓN")
        print("="*60)
        
        # 12. Estandarizar variables numéricas
        X_train, X_test, scaler = scale_train_test(X_train, X_test)

        # 13. Guardar datos procesados
        save_processed_data(X_train, X_test, y_train, y_test, scaler)
        
        print("\n" + "="*60)
        print("✅ PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"📊 Características finales: {X_train.shape[1]}")
        print(f"📈 Muestras de entrenamiento: {X_train.shape[0]:,}")
        print(f"📉 Muestras de prueba: {X_test.shape[0]:,}")
        print(f"🎯 Reducción de dimensionalidad lograda!")
        print("="*60)
        
        return X_train, X_test, y_train, y_test, scaler
    
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR EN EL PREPROCESAMIENTO")
        print("="*60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Ejecutar con configuración optimizada para alto rendimiento
    # Ajusta estos parámetros según tus necesidades
    X_train, X_test, y_train, y_test, scaler = main(
        correlation_threshold=0.90,      # Más agresivo: eliminar características con correlación > 0.90
        variance_threshold=0.01,         # Eliminar características con varianza < 0.01
        max_categories=15                # Máximo 15 categorías por variable categórica
    )