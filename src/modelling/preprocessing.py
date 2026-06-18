import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

TARGET_COLUMN = "not.fully.paid"


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa features y target, y convierte la variable categórica 'purpose'
    en variables dummy (one-hot encoding).

    Args:
        df: dataset cargado por load_loan_dataset.

    Returns:
        Tupla (X, y) lista para el split train/test.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # One-hot encoding de 'purpose' — drop_first evita colinealidad perfecta,
    # relevante porque Logistic Regression es sensible a eso
    X = pd.get_dummies(X, columns=["purpose"], drop_first=True)

    logger.info(f"Features preparadas: {X.shape[1]} columnas tras encoding")

    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split estratificado train/test — mantiene la proporción de impagos
    en ambos conjuntos, crítico para evaluar bien Gini y KS después.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y    # esto es lo que garantiza la misma proporción de clases
    )

    logger.info(
        f"Split realizado — train: {len(X_train)} ({y_train.mean():.2%} impago), "
        f"test: {len(X_test)} ({y_test.mean():.2%} impago)"
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Escala las features numéricas. Necesario para Logistic Regression
    (no para XGBoost, que es invariante a escala, pero no estorba).

    IMPORTANTE: el scaler se ajusta SOLO con train y se aplica a test —
    nunca al revés, o hay data leakage del test set.
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler