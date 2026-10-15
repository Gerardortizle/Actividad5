"""
datos_prep.py — Funciones de limpieza, estandarización, validación y versionado
del dataset temporal SKU–sucursal para DEMANDA-IA (Farmacias Occidente Salud, Fase I).

Cada regla de limpieza tiene un criterio explícito (ver REGLAS) y queda contabilizada
en un reporte de calidad (JSON) para que el proceso sea auditable y reproducible.

Uso desde consola (desde la raíz del repositorio):
    python fuentes/datos_prep.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_INI = RAIZ / "datos" / "datos_ini" / "train.csv"
RUTA_LIMP = RAIZ / "datos" / "datos_limp" / "ventas_sku_sucursal_limpio.csv.gz"
RUTA_REPORTE = RAIZ / "datos" / "datos_limp" / "reporte_calidad.json"
RUTA_MANIFIESTO = RAIZ / "datos" / "MANIFIESTO_DATOS.json"

VERSION_DATOS = "1.1.0"          # versión semántica del dataset limpio
COLUMNAS_ORIGEN = {"date": "fecha", "store": "sucursal_id", "item": "sku_id", "sales": "unidades"}
LLAVE = ["fecha", "sucursal_id", "sku_id"]
SERIE = ["sucursal_id", "sku_id"]
Z_ANOMALIA = 3.5                  # umbral de z-score robusto (Iglewicz y Hoaglin)

REGLAS = {
    "R1_esquema": "Renombrar columnas a español snake_case y forzar tipos (fecha=datetime, ids=int, unidades=float).",
    "R2_llave_invalida": "Eliminar registros con fecha no parseable o ids nulos/no positivos: no pueden atribuirse a una serie.",
    "R3_duplicados_exactos": "Eliminar filas idénticas (doble carga del mismo archivo POS).",
    "R4_duplicados_llave": "Si una misma fecha-sucursal-SKU tiene valores distintos, conservar el último registro cargado (corrección posterior del POS).",
    "R5_negativos": "Unidades < 0 son devoluciones, no demanda: se acotan a 0 y se marcan.",
    "R6_calendario": "Completar el calendario diario de cada serie; los días faltantes se tratan como dato faltante (no como venta cero).",
    "R7_imputacion": "Imputar faltantes con la mediana de la misma serie y mismo día de la semana en las 4 semanas previas (respaldo: mediana de la serie). Se marca con es_imputado=1.",
    "R8_anomalias": f"Marcar (no eliminar) observaciones con |z robusto| > {Z_ANOMALIA} respecto a la mediana móvil centrada de 28 días; un pico real de demanda (p. ej., temporada respiratoria) es información.",
    "R9_codigos": "Generar códigos estándar SUC-XX y SKU-XXX para integración con el ERP.",
}


# ---------------------------------------------------------------------------
# Utilidades de versionado
# ---------------------------------------------------------------------------
def sha256_archivo(ruta: Path, bloque: int = 1 << 20) -> str:
    """Huella SHA-256 del archivo: identifica de forma única la versión de los datos."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def sha256_dataframe(df: pd.DataFrame) -> str:
    """Huella del contenido lógico (independiente de la compresión del archivo)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Limpieza y estandarización
# ---------------------------------------------------------------------------
def cargar_datos_ini(ruta: Path = RUTA_INI) -> pd.DataFrame:
    return pd.read_csv(ruta)


def estandarizar_esquema(df: pd.DataFrame, reporte: dict) -> pd.DataFrame:
    """R1 + R2: nombres, tipos y llaves válidas."""
    df = df.rename(columns=COLUMNAS_ORIGEN).copy()
    reporte["filas_iniciales"] = int(len(df))
    reporte["nulos_por_columna_inicial"] = {c: int(v) for c, v in df.isna().sum().items()}

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for c in SERIE:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["unidades"] = pd.to_numeric(df["unidades"], errors="coerce").astype("float64")

    invalida = df["fecha"].isna() | df["sucursal_id"].isna() | df["sku_id"].isna() \
        | (df["sucursal_id"] <= 0) | (df["sku_id"] <= 0)
    reporte["R2_llave_invalida_eliminadas"] = int(invalida.sum())
    df = df.loc[~invalida].copy()
    df[SERIE] = df[SERIE].astype("int16")
    return df


def eliminar_duplicados(df: pd.DataFrame, reporte: dict) -> pd.DataFrame:
    """R3 + R4."""
    n0 = len(df)
    df = df.drop_duplicates()
    reporte["R3_duplicados_exactos_eliminados"] = int(n0 - len(df))
    n1 = len(df)
    df = df.drop_duplicates(subset=LLAVE, keep="last")
    reporte["R4_duplicados_llave_resueltos"] = int(n1 - len(df))
    return df


def aplicar_reglas_negocio(df: pd.DataFrame, reporte: dict) -> pd.DataFrame:
    """R5: devoluciones no son demanda."""
    neg = df["unidades"] < 0
    reporte["R5_negativos_acotados"] = int(neg.sum())
    df["es_devolucion"] = neg.astype("int8")
    df.loc[neg, "unidades"] = 0.0
    return df


def completar_calendario(df: pd.DataFrame, reporte: dict) -> pd.DataFrame:
    """R6: rejilla diaria completa por serie (fecha mínima a máxima global)."""
    fechas = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
    series = df[SERIE].drop_duplicates()
    rejilla = series.merge(pd.DataFrame({"fecha": fechas}), how="cross")
    n_antes = len(df)
    df = rejilla.merge(df, on=["sucursal_id", "sku_id", "fecha"], how="left")
    reporte["R6_dias_faltantes_agregados"] = int(len(df) - n_antes)
    df["es_devolucion"] = df["es_devolucion"].fillna(0).astype("int8")
    return df.sort_values(SERIE + ["fecha"]).reset_index(drop=True)


def imputar_faltantes(df: pd.DataFrame, reporte: dict) -> pd.DataFrame:
    """R7: mediana de la misma serie y día de la semana (4 semanas previas)."""
    falt = df["unidades"].isna()
    reporte["R7_valores_imputados"] = int(falt.sum())
    df["es_imputado"] = falt.astype("int8")
    if falt.any():
        dow = df["fecha"].dt.dayofweek
        med_dow = df.groupby(SERIE + [dow])["unidades"].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).median())
        med_serie = df.groupby(SERIE)["unidades"].transform("median")
        df["unidades"] = df["unidades"].fillna(med_dow).fillna(med_serie).fillna(0.0)
    return df


def marcar_anomalias(df: pd.DataFrame, reporte: dict) -> pd.DataFrame:
    """R8: z-score robusto sobre el residuo contra la mediana móvil centrada (28 días)."""
    g = df.groupby(SERIE)["unidades"]
    mediana_movil = g.transform(lambda s: s.rolling(28, center=True, min_periods=7).median())
    resid = df["unidades"] - mediana_movil
    mad = resid.abs().groupby([df["sucursal_id"], df["sku_id"]]).transform("median")
    z = 0.6745 * resid / mad.replace(0, np.nan)
    df["z_robusto"] = z.astype("float32")
    df["es_anomalia"] = (z.abs() > Z_ANOMALIA).fillna(False).astype("int8")
    reporte["R8_anomalias_marcadas"] = int(df["es_anomalia"].sum())
    reporte["R8_pct_anomalias"] = round(100 * df["es_anomalia"].mean(), 4)
    return df


def generar_codigos(df: pd.DataFrame) -> pd.DataFrame:
    """R9: códigos estándar para el ERP."""
    df["codigo_sucursal"] = "SUC-" + df["sucursal_id"].astype(str).str.zfill(2)
    df["codigo_sku"] = "SKU-" + df["sku_id"].astype(str).str.zfill(3)
    df["unidades"] = df["unidades"].round().astype("int32")
    cols = ["fecha", "sucursal_id", "sku_id", "codigo_sucursal", "codigo_sku", "unidades",
            "es_imputado", "es_devolucion", "es_anomalia", "z_robusto"]
    return df[cols]


def pipeline_limpieza(df_ini: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica R1–R9 en orden y devuelve (datos_limpios, reporte_de_calidad)."""
    rep: dict = {}
    df = estandarizar_esquema(df_ini, rep)
    df = eliminar_duplicados(df, rep)
    df = aplicar_reglas_negocio(df, rep)
    df = completar_calendario(df, rep)
    df = imputar_faltantes(df, rep)
    df = marcar_anomalias(df, rep)
    df = generar_codigos(df)
    rep.update({
        "filas_finales": int(len(df)),
        "n_sucursales": int(df["sucursal_id"].nunique()),
        "n_skus": int(df["sku_id"].nunique()),
        "n_series": int(df.groupby(SERIE).ngroups),
        "fecha_min": str(df["fecha"].min().date()),
        "fecha_max": str(df["fecha"].max().date()),
        "nulos_finales": int(df.isna().sum().sum() - df["z_robusto"].isna().sum()),
        "reglas": REGLAS,
    })
    return df, rep


# ---------------------------------------------------------------------------
# Ingeniería de variables (sin fuga de información)
# ---------------------------------------------------------------------------
HORIZONTE = 7  # días de anticipación: el pronóstico nocturno debe cubrir el lead time de reabasto

FEATURES_NUM = ["lag_7", "lag_14", "lag_21", "lag_28", "lag_364",
                "media_7", "media_28", "std_28", "media_dow_4", "tendencia"]
FEATURES_CAL = ["dia_semana", "mes", "dia_mes", "semana_anio", "es_fin_semana"]
FEATURES_ID = ["sucursal_id", "sku_id"]
FEATURES = FEATURES_ID + FEATURES_CAL + FEATURES_NUM
OBJETIVO = "unidades"


def construir_features(df: pd.DataFrame, horizonte: int = HORIZONTE) -> pd.DataFrame:
    """Todas las variables usan información de al menos `horizonte` días antes."""
    df = df.sort_values(SERIE + ["fecha"]).copy()
    g = df.groupby(SERIE)["unidades"]
    for l in (7, 14, 21, 28, 364):
        df[f"lag_{l}"] = g.shift(l).astype("float32")
    base = g.shift(horizonte)
    gb = base.groupby([df["sucursal_id"], df["sku_id"]])
    df["media_7"] = gb.transform(lambda s: s.rolling(7).mean()).astype("float32")
    df["media_28"] = gb.transform(lambda s: s.rolling(28).mean()).astype("float32")
    df["std_28"] = gb.transform(lambda s: s.rolling(28).std()).astype("float32")
    df["media_dow_4"] = ((df["lag_7"] + df["lag_14"] + df["lag_21"] + df["lag_28"]) / 4).astype("float32")
    df["dia_semana"] = df["fecha"].dt.dayofweek.astype("int8")
    df["mes"] = df["fecha"].dt.month.astype("int8")
    df["dia_mes"] = df["fecha"].dt.day.astype("int8")
    df["semana_anio"] = df["fecha"].dt.isocalendar().week.astype("int8")
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["tendencia"] = ((df["fecha"] - pd.Timestamp("2013-01-01")).dt.days / 365.25).astype("float32")
    return df.dropna(subset=FEATURES).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def main() -> None:
    df_ini = cargar_datos_ini()
    df, rep = pipeline_limpieza(df_ini)
    RUTA_LIMP.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_LIMP, index=False, compression={"method": "gzip", "mtime": 0})

    rep["version_datos"] = VERSION_DATOS
    rep["generado_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rep["sha256_datos_ini"] = sha256_archivo(RUTA_INI)
    rep["sha256_contenido_limpio"] = sha256_dataframe(df)
    with open(RUTA_REPORTE, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    manifiesto = {
        "datos_ini/train.csv": {"version": "1.0.0", "sha256": rep["sha256_datos_ini"],
                                 "filas": rep["filas_iniciales"],
                                 "fuente": "Kaggle — Store Item Demand Forecasting Challenge (train.csv)"},
        "datos_limp/ventas_sku_sucursal_limpio.csv.gz": {
            "version": VERSION_DATOS, "sha256_contenido": rep["sha256_contenido_limpio"],
            "filas": rep["filas_finales"], "generado_por": "fuentes/datos_prep.py"},
    }
    with open(RUTA_MANIFIESTO, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "reglas"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
