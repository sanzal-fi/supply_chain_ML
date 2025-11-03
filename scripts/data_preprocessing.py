"""
Script de Preprocesamiento de Datos
====================================

Este script contiene todas las funciones necesarias para preprocesar
el dataset DataCoSupplyChainDataset.csv para el modelado de predicción
de demoras en entregas.

Autor: Sistema de Predicción de Demoras
Fecha: 2024
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    
    df = None
    for encoding in encodings:
        try:
            print(f"Probando codificación: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✅ Éxito con codificación: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"❌ Error con {encoding}")
            continue
        except Exception as e:
            print(f"❌ Error inesperado con {encoding}: {str(e)[:50]}...")
            continue
    
    if df is None:
        print("❌ No se pudo cargar el archivo con ninguna codificación estándar")
        print("Intentando con encoding='latin-1' y manejo de errores...")
        df = pd.read_csv(file_path, encoding='latin-1', encoding_errors='replace')
    
    print(f"Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
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
    df['demora'] = (df['Days for shipping (real)'] > df['Days for shipment (scheduled)']).astype(int)
    
    # Mostrar distribución
    demora_0 = sum(df['demora'] == 0)
    demora_1 = sum(df['demora'] == 1)
    total = len(df)
    
    print(f"Entregas a tiempo (demora=0): {demora_0:,} ({demora_0/total*100:.1f}%)")
    print(f"Entregas con demora (demora=1): {demora_1:,} ({demora_1/total*100:.1f}%)")
    
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
        # Sensibles/irrelevantes
        'Customer Password', 'Customer Email', 'Customer Fname', 
        'Customer Lname', 'Customer Street', 'Product Description', 
        'Product Image',
        # Fuga directa de información (conocidas solo después del resultado)
        'Days for shipping (real)',
        'Late_delivery_risk',
        'Order Status',
        'Delivery Status',
        'Shipping Date (Actual)',
        # Campos de fecha reales posteriores al envío
        'Shipping Date (DateOrders)',  # por si existe esta variante
        # Identificadores de alta cardinalidad (no predictivos)
        'Order Id', 'Order Item Id', 'Order Item Cardprod Id', 'Product Card Id',
        'Product Category Id', 'Customer Id', 'Order Customer Id'
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

def reduce_cardinality(df, max_categories=50):
    """
    Reducir la cardinalidad de variables categóricas agrupando categorías raras en 'Other'.

    Args:
        df (pd.DataFrame): Dataset limpio
        max_categories (int): Número máximo de categorías a conservar por columna

    Returns:
        pd.DataFrame: Dataset con cardinalidad reducida
    """
    print("Reduciendo cardinalidad de variables categóricas...")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        value_counts = df[col].value_counts(dropna=False)
        if len(value_counts) > max_categories:
            top_categories = value_counts.index[:max_categories]
            original_unique = len(value_counts)
            df[col] = df[col].where(df[col].isin(top_categories), other='Other')
            reduced_unique = df[col].nunique(dropna=False)
            print(f"  {col}: {original_unique} -> {reduced_unique} categorías (agrupado 'Other')")
    return df

def handle_outliers(df):
    """
    Tratar outliers en variables numéricas usando el método IQR.
    
    Args:
        df (pd.DataFrame): Dataset limpio
        
    Returns:
        pd.DataFrame: Dataset con outliers tratados
    """
    print("Tratando outliers...")
    
    # Variables numéricas para análisis de outliers
    numeric_cols = ['Sales', 'Order Item Quantity', 'Benefit per order', 
                    'Days for shipping (real)', 'Days for shipment (scheduled)',
                    'Order Item Product Price', 'Order Item Total']
    
    # Filtrar solo las que existen
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    outliers_removed = 0
    
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
            # Cap outliers en lugar de eliminarlos
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            outliers_removed += outlier_count
            print(f"  {col}: {outlier_count:,} outliers capados")
    
    print(f"Total de outliers tratados: {outliers_removed:,}")
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
    
    print(f"Variables categóricas a codificar: {len(categorical_cols)}")
    
    # Aplicar one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    
    print(f"Variables después del encoding: {df_encoded.shape[1]}")
    print(f"Variables agregadas: {df_encoded.shape[1] - df.shape[1]}")
    
    return df_encoded

def scale_features(df):
    """
    Estandarizar variables numéricas usando StandardScaler.
    
    Args:
        df (pd.DataFrame): Dataset con variables codificadas
        
    Returns:
        tuple: (df_scaled, scaler) - Dataset escalado y objeto scaler
    """
    print("Estandarizando variables numéricas...")
    
    # Identificar variables numéricas (excluyendo la variable objetivo)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'demora' in numeric_cols:
        numeric_cols.remove('demora')
    
    print(f"Variables numéricas a escalar: {len(numeric_cols)}")
    
    # Crear y ajustar scaler (evitar copias grandes)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Variables numéricas escaladas correctamente")
    return df, scaler

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
    
    # Guardar scaler
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    # Guardar nombres de características
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, f'{output_dir}/feature_names.pkl')
    
    print(f"Datos guardados en: {output_dir}")
    print("Archivos guardados:")
    print("  - X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("  - scaler.pkl, feature_names.pkl")

def main():
    """
    Función principal que ejecuta todo el pipeline de preprocesamiento.
    """
    print("="*60)
    print("PIPELINE DE PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    # 1. Cargar datos (resolver ruta desde la ubicación del script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    csv_path = os.path.join(project_root, 'DataCoSupplyChainDataset.csv')
    df = load_data(csv_path)
    
    # 2. Crear variable objetivo
    df = create_target_variable(df)
    
    # 3. Limpiar datos
    df = clean_data(df)
    
    # 4. Tratar outliers
    df = handle_outliers(df)
    
    # 5. Reducir cardinalidad de categóricas antes del one-hot
    df = reduce_cardinality(df, max_categories=50)

    # 6. Codificar variables categóricas
    df = encode_categorical(df)

    # 7. Dividir datos (antes de escalar para evitar fuga)
    X_train, X_test, y_train, y_test = split_data(df)

    # 8. Estandarizar variables numéricas SOLO con información de TRAIN
    X_train, X_test, scaler = scale_train_test(X_train, X_test)

    # 9. Guardar datos procesados
    save_processed_data(X_train, X_test, y_train, y_test, scaler)
    
    print("\n" + "="*60)
    print("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = main()
