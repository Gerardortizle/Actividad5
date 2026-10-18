"""
Project : Act 5. Baseline de pronóstico SKU–sucursal JGOL

    Name: José Gerardo Ortiz Leñero
    Matricula: AL07182099
    Date: Octubre 18 2026
    Universidad TecMilenio
    
Prueba de robustez: el pipeline debe detectar y corregir cada tipo de error
que produce una exportación real de punto de venta (POS). Ejecutar: python pruebas/test_datos_prep.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fuentes"))
from datos_prep import pipeline_limpieza


def datos_contaminados() -> pd.DataFrame:
    fechas = pd.date_range("2017-01-01", periods=60, freq="D")
    rng = np.random.default_rng(42)
    df = pd.concat([pd.DataFrame({"date": fechas.strftime("%Y-%m-%d"), "store": s, "item": i,
                                  "sales": rng.poisson(30, len(fechas)).astype(float)})
                    for s, i in [(1, 1), (2, 7)]], ignore_index=True)
    df = pd.concat([df, df.iloc[[3, 4]]])                                  # 2 duplicados exactos
    conflicto = df.iloc[[10]].copy(); conflicto["sales"] = 99              # 1 duplicado de llave
    df = pd.concat([df, conflicto])
    df.loc[df.index[20], "sales"] = np.nan                                 # 1 nulo
    df.loc[df.index[25], "sales"] = -4                                     # 1 devolución
    df = df.drop(index=df.index[[30, 31]])                                 # 2 días faltantes
    extra = pd.DataFrame({"date": ["fecha-mala", "2017-01-05"], "store": [1, None], "item": [1, 1], "sales": [5, 5]})
    df = pd.concat([df, extra], ignore_index=True)                         # 2 llaves inválidas
    df.loc[df.index[45], "sales"] = 400                                    # 1 pico anómalo
    return df


if __name__ == "__main__":
    limpio, rep = pipeline_limpieza(datos_contaminados())
    assert rep["R2_llave_invalida_eliminadas"] == 2, rep
    assert rep["R3_duplicados_exactos_eliminados"] == 2, rep
    assert rep["R4_duplicados_llave_resueltos"] == 1, rep
    assert rep["R5_negativos_acotados"] == 1, rep
    assert rep["R6_dias_faltantes_agregados"] == 2, rep
    assert rep["R7_valores_imputados"] == 3, rep          # 1 nulo + 2 días faltantes
    assert rep["R8_anomalias_marcadas"] >= 1, rep
    assert len(limpio) == 120 and limpio["unidades"].isna().sum() == 0
    assert limpio.duplicated(["fecha", "sucursal_id", "sku_id"]).sum() == 0
    print("OK — todas las reglas R2–R8 se activaron como se esperaba")
    for k, v in rep.items():
        if k.startswith("R"):
            print(f"  {k}: {v}")
