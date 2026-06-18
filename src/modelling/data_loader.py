import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "credit.policy", "purpose", "int.rate", "installment", "log.annual.inc",
    "dti", "fico", "days.with.cr.line", "revol.bal", "revol.util",
    "inq.last.6mths", "delinq.2yrs", "pub.rec", "not.fully.paid"
]

TARGET_COLUMN = "not.fully.paid"


def load_loan_dataset(path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset de Lending Club (versión 'Loan Data' de Kaggle) y valida
    que tiene la estructura esperada para un modelo de PD.

    Args:
        path: ruta al CSV del dataset.

    Returns:
        DataFrame validado y con tipos básicos corregidos.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si faltan columnas esperadas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    df = pd.read_csv(path)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {missing}")

    # Tipos explícitos — evita sorpresas si el CSV trae algo como string en vez de int
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df["purpose"] = df["purpose"].astype("category")

    logger.info(f"Dataset cargado: {len(df)} filas, {df[TARGET_COLUMN].mean():.2%} tasa de impago")

    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Genera un resumen del dataset para que el LLM lo use al redactar el informe
    del modelo — evita que el LLM tenga que adivinar estadísticas descriptivas.
    """
    return {
        "n_rows": len(df),
        "default_rate": round(df[TARGET_COLUMN].mean(), 4),
        "n_features": len(df.columns) - 1,
        "purpose_distribution": df["purpose"].value_counts().to_dict(),
        "fico_range": [int(df["fico"].min()), int(df["fico"].max())],
    }