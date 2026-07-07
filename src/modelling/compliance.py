"""
Verificaciones deterministas y objetivas sobre el dataset usado para modelado de PD.

Filosofía: este módulo NO decide si algo "cumple la normativa" en sentido legal,
ni cita artículos — eso es responsabilidad del LLM, combinando esta salida con los
resultados de search_regulation. Aquí solo se comprueban HECHOS OBJETIVOS sobre los
datos (tamaño muestral, missing values, balance de clases, variables sensibles,
trazabilidad temporal) de forma fija y reproducible, igual que el resto de
src/modelling/.
"""

import pandas as pd

MIN_SAMPLE_SIZE = 1000
MAX_MISSING_RATIO = 0.20
MIN_DEFAULT_RATE = 0.01
MAX_DEFAULT_RATE = 0.50

# Variables que los principios de no discriminación / fair lending (p.ej. EBA
# GL/2017/16) desaconsejan usar directamente como input de un modelo de scoring
SENSITIVE_VARIABLE_KEYWORDS = [
    "sex", "gender", "genero", "sexo", "race", "raza", "ethnic", "etnia",
    "religion", "nationality", "nacionalidad", "marital", "estado_civil",
]

# Columnas que indicarían info temporal explotable para verificar el periodo
# mínimo de observación histórica exigido normativamente
DATE_COLUMN_KEYWORDS = ["date", "fecha", "year", "año", "time", "period"]


def check_sample_size(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "check": "tamaño_muestral",
        "status": "cumple" if n >= MIN_SAMPLE_SIZE else "no_cumple",
        "detail": f"{n} observaciones (mínimo recomendado: {MIN_SAMPLE_SIZE})",
    }


def check_target_definition(df: pd.DataFrame, target_column: str) -> dict:
    if target_column not in df.columns:
        return {
            "check": "definición_variable_objetivo",
            "status": "no_cumple",
            "detail": f"No se encuentra la columna objetivo '{target_column}'",
        }
    nulls = int(df[target_column].isna().sum())
    unique_vals = df[target_column].dropna().unique()
    is_binary = set(unique_vals).issubset({0, 1})
    status = "cumple" if (nulls == 0 and is_binary) else "no_cumple"
    return {
        "check": "definición_variable_objetivo",
        "status": status,
        "detail": f"Valores únicos: {sorted(unique_vals)}, nulos: {nulls}",
    }


def check_default_rate(df: pd.DataFrame, target_column: str) -> dict:
    rate = df[target_column].mean()
    status = "cumple" if MIN_DEFAULT_RATE <= rate <= MAX_DEFAULT_RATE else "no_cumple"
    return {
        "check": "tasa_de_default_razonable",
        "status": status,
        "detail": f"Tasa de impago observada: {rate:.2%}",
    }


def check_missing_values(df: pd.DataFrame) -> dict:
    ratios = df.isna().mean()
    offending = ratios[ratios > MAX_MISSING_RATIO]
    status = "cumple" if offending.empty else "no_cumple"
    detail = (
        "Sin columnas con missing values excesivos"
        if offending.empty
        else f"Columnas con >{MAX_MISSING_RATIO:.0%} missing: {dict(offending.round(3))}"
    )
    return {"check": "missing_values", "status": status, "detail": detail}


def check_sensitive_variables(df: pd.DataFrame) -> dict:
    found = [c for c in df.columns if any(k in c.lower() for k in SENSITIVE_VARIABLE_KEYWORDS)]
    status = "alerta" if found else "cumple"
    detail = (
        f"Variables potencialmente sensibles detectadas: {found} — revisar "
        f"principios de no discriminación antes de usarlas como input directo"
        if found
        else "No se detectan variables sensibles obvias en los nombres de columna"
    )
    return {"check": "variables_sensibles_no_discriminacion", "status": status, "detail": detail}


def check_temporal_traceability(df: pd.DataFrame) -> dict:
    found = [c for c in df.columns if any(k in c.lower() for k in DATE_COLUMN_KEYWORDS)]
    status = "cumple" if found else "no_verificable"
    detail = (
        f"Columnas temporales encontradas: {found}"
        if found
        else (
            "No hay columna de fecha/periodo en el dataset — no se puede verificar "
            "el periodo mínimo de observación histórica exigido por la normativa "
            "(p. ej. CRR Art. 180) con estos datos"
        )
    )
    return {"check": "trazabilidad_temporal", "status": status, "detail": detail}


def check_dataset_compliance(df: pd.DataFrame, target_column: str) -> dict:
    """
    Ejecuta todas las verificaciones deterministas y devuelve un resumen
    estructurado. El LLM debe combinar esta salida con las citas normativas
    obtenidas vía search_regulation para redactar la valoración final — esta
    función no contiene ningún texto legal ni cita ningún artículo.
    """
    checks = [
        check_sample_size(df),
        check_target_definition(df, target_column),
        check_default_rate(df, target_column),
        check_missing_values(df),
        check_sensitive_variables(df),
        check_temporal_traceability(df),
    ]
    resumen = {
        "cumple": sum(1 for c in checks if c["status"] == "cumple"),
        "no_cumple": sum(1 for c in checks if c["status"] == "no_cumple"),
        "alerta_o_no_verificable": sum(
            1 for c in checks if c["status"] in ("alerta", "no_verificable")
        ),
        "total": len(checks),
    }
    return {"checks": checks, "resumen": resumen}