"""Инференс лучшей модели (XGBoost Tuned) / Inference for the best model (XGBoost Tuned)."""

from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd

# Корень проекта: src/../ / Project root: parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


def _sanitize_columns(columns):
    """XGBoost не принимает [, ] и < в именах признаков / XGBoost forbids [, ] and < in feature names."""
    return [str(c).replace("[", "(").replace("]", ")").replace("<", "lt") for c in columns]


def load_model(model_path: Union[Path, str] = MODEL_PATH):
    """Загружает артефакт из models/best_model.pkl / Load the artifact from models/best_model.pkl.

    Returns:
        dict: model, scaler, feature_names и метаданные предобработки /
              model, scaler, feature_names and preprocessing metadata.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Модель не найдена / Model not found: {path}. "
            "Сначала выполните ячейку сохранения в ноутбуке / "
            "Run the notebook save cell first."
        )
    return joblib.load(path)


def preprocess_input(raw_data, artifact: Optional[dict] = None) -> pd.DataFrame:
    """Применяет те же преобразования, что в ноутбуке / Apply the same transforms as in the notebook.

    Args:
        raw_data: dict, list[dict] или DataFrame с сырыми полями датчиков /
                  dict, list[dict], or DataFrame with raw sensor fields.
        artifact: результат load_model(); если None — загружается с диска /
                  load_model() result; loaded from disk if None.
    """
    if artifact is None:
        artifact = load_model()

    if isinstance(raw_data, dict):
        df = pd.DataFrame([raw_data])
    elif isinstance(raw_data, list):
        df = pd.DataFrame(raw_data)
    elif isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    else:
        raise TypeError("raw_data должен быть dict, list[dict] или DataFrame / raw_data must be dict, list[dict], or DataFrame")

    product_id_col = artifact["product_id_col"]
    prefix_col = artifact["product_id_prefix_col"]
    prefix_len = artifact["product_id_prefix_len"]
    numeric_features = artifact["numeric_features"]
    categorical_features = artifact["categorical_features"]
    drop_cols = artifact["drop_cols"]
    feature_names = artifact["feature_names"]
    scaler = artifact["scaler"]

    # Префикс Product ID (L4, M1, H3, ...) / Product ID prefix
    if product_id_col in df.columns:
        df[prefix_col] = df[product_id_col].astype(str).str[:prefix_len]

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    missing_numeric = [c for c in numeric_features if c not in X.columns]
    if missing_numeric:
        raise ValueError(f"Нет числовых колонок / Missing numeric columns: {missing_numeric}")

    missing_cat = [c for c in categorical_features if c not in X.columns]
    if missing_cat:
        raise ValueError(f"Нет категориальных колонок / Missing categorical columns: {missing_cat}")

    # One-Hot Encoding, как pd.get_dummies в ноутбуке / Same OHE as in the notebook
    X = pd.get_dummies(X, columns=categorical_features, drop_first=False)

    # Стандартизация только числовых признаков / Scale numeric features only
    X[numeric_features] = scaler.transform(X[numeric_features])

    # Те же имена колонок, что у X_train / Same column names as X_train
    X.columns = _sanitize_columns(X.columns)
    X = X.reindex(columns=feature_names, fill_value=0)
    return X


def predict(raw_data, artifact: Optional[dict] = None):
    """Возвращает класс (0/1) и вероятность отказа / Return class (0/1) and failure probability.

    Для одной строки — (int, float); для нескольких — (list[int], list[float]).
    For one row — (int, float); for several rows — (list[int], list[float]).
    """
    if artifact is None:
        artifact = load_model()

    X = preprocess_input(raw_data, artifact=artifact)
    model = artifact["model"]
    preds = model.predict(X)
    # вероятность класса 1 (отказ) / probability of class 1 (failure)
    proba = model.predict_proba(X)[:, 1]

    if len(preds) == 1:
        return int(preds[0]), float(proba[0])
    return [int(p) for p in preds], [float(p) for p in proba]


if __name__ == "__main__":
    # Демо-запись в формате сырого CSV / Demo record in raw CSV format
    sample = {
        "UDI": 1,
        "Product ID": "M14860",
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }

    loaded = load_model()
    pred, prob = predict(sample, artifact=loaded)
    print("Демо-предсказание / Demo prediction")
    print(f"  class (0 = no failure, 1 = failure): {pred}")
    print(f"  P(failure): {prob:.4f}")
