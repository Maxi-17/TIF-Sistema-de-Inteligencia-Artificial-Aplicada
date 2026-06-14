from pathlib import Path
import joblib
import pandas as pd
import numpy as np


def load_model_and_preprocessor(models_dir: Path | None = None):
    """Carga el modelo y el preprocessor desde la carpeta `models`.

    Devuelve una tupla (model, preprocessor). Si alguno no existe, devuelve None en su lugar.
    """
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
    model = None
    preprocessor = None
    try:
        mpath = models_dir / "modelo_ganador.pkl"
        if mpath.exists():
            model = joblib.load(mpath)
    except Exception:
        model = None
    try:
        ppath = models_dir / "preprocessor.pkl"
        if ppath.exists():
            preprocessor = joblib.load(ppath)
    except Exception:
        preprocessor = None
    return model, preprocessor


def prepare_input_dataframe(inputs: dict, expected_columns: list = None, categorical_columns: list = None):
    """Construye un DataFrame a partir de `inputs`.

    Si 'expected_columns' está provisto, se crea un DataFrame que contiene exactamente
    esas columnas (rellenando con 0s cuando falta informacion). En caso contrario,
    se crea un DataFrame minimal con las columnas principales usadas en la demo.
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Si el usuario nos pasa la lista de columnas esperadas, construimos el df acorde
    if expected_columns is not None:
        row = {}
        sel_month = inputs.get('Month', None)
        browser_value = int(inputs.get('Browser', 12) or 12)
        traffic_type_value = int(inputs.get('TrafficType', 1) or 1)
        for col in expected_columns:
            if col == 'PageValues':
                row[col] = float(inputs.get('PageValues', 0.0))
            elif col == 'Administrative_Duration':
                row[col] = float(inputs.get('Administrative_Duration', 0.0))
            elif col == 'ProductRelated_Duration':
                row[col] = float(inputs.get('ProductRelated_Duration', 0.0))
            elif col == 'TrafficType':
                row[col] = traffic_type_value
            elif col.startswith('TrafficType_'):
                try:
                    row[col] = 1 if int(col.split('_', 1)[1]) == traffic_type_value else 0
                except Exception:
                    row[col] = 0
            elif col == 'Browser':
                row[col] = browser_value
            elif col == 'Browser_12':
                row[col] = int(browser_value == 12)
            elif col == 'Month':
                row[col] = str(inputs.get('Month', ''))
            elif col.startswith('Month_'):
                if sel_month is not None and col.lower().endswith(sel_month.lower()[:3]):
                    row[col] = 1
                else:
                    row[col] = 0
            else:
                if categorical_columns and col in categorical_columns:
                    row[col] = str(inputs.get(col, ''))
                else:
                    # Rellenar con el valor del diccionario si existe, sino 0
                    row[col] = inputs.get(col, 0)
        return pd.DataFrame([row])

    # Comportamiento por defecto (mini esquema) si no se conocen las columnas esperadas
    data = {}
    for k, v in inputs.items():
        data[k] = v
    return pd.DataFrame([data])


def predict_from_inputs(model, preprocessor, inputs_dict):
    """Recibe un dict de inputs, aplica el preprocessor si existe y devuelve (pred, prob).

    - pred: clase predicha (0 o 1)
    - prob: probabilidad de clase positiva (float entre 0 y 1)
    """
    # Determinar si el modelo es un Pipeline integrado
    model_contains_preproc = False
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        if isinstance(model, Pipeline):
            for _name, step in model.steps:
                if isinstance(step, ColumnTransformer):
                    model_contains_preproc = True
                    break
    except Exception:
        model_contains_preproc = False

    # Extraer nombres de columnas esperadas para el alineamiento de datos
    expected = None
    cat_cols = []
    if preprocessor is not None and hasattr(preprocessor, 'feature_names_in_'):
        expected = list(preprocessor.feature_names_in_)
    elif model is not None and hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)

    # Construir el DataFrame con las columnas y orden correctos
    if expected is not None:
        df = prepare_input_dataframe(inputs_dict, expected_columns=expected, categorical_columns=cat_cols)
    else:
        df = prepare_input_dataframe(inputs_dict)

    # Procesar transformacion
    if preprocessor is not None and not model_contains_preproc:
        model_input = preprocessor.transform(df)
    else:
        model_input = df

    # Ejecutar la inferencia probabilistica de forma segura para el editor
    if model is not None and hasattr(model, 'predict_proba'):
        probs = model.predict_proba(model_input)
        prob = float(probs[0][1])
        pred = int(prob >= 0.5)
    elif model is not None:
        pred = int(model.predict(model_input)[0])
        prob = float(pred)
    else:
        # Resguardo absoluto por si de verdad el modelo no cargo
        pred, prob = 0, 0.0

    # CORRECCION CLAVE: Devolver (prediccion, probabilidad)
    return pred, prob