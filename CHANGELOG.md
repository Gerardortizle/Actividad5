# CHANGELOG — Actividad 5 · DEMANDA-IA Fase I

Formato basado en *Keep a Changelog* y versionado semántico. Las versiones de **datos** y de **modelo** se etiquetan en Git; los hashes SHA-256 están en `datos/MANIFIESTO_DATOS.json`.

## [modelo-v1.2.0] — 2026-10-15 · Baseline entrenado y registrado
### Agregado
- Ejecución completa de `fuentes/train.py` desde `entrena.ipynb` (commit de código `b43a8b8`), con 30 corridas en el experimento `DEMANDA-IA_FaseI_baseline_SKU_sucursal`.
- Grid Search con validación temporal de 3 folds para tres algoritmos:
  - Ridge: 5 combinaciones. Ganó α = 100, aunque el resultado es prácticamente insensible a α (MAE CV 6.463–6.467).
  - Random Forest: 8 combinaciones. Ganó `max_depth=16, max_features=0.5, min_samples_leaf=10`.
  - XGBoost: 12 combinaciones. Ganó `learning_rate=0.05, max_depth=6, objective=reg:squarederror`.
- Resultados en el holdout jul–dic 2017 (MAE):
  - naive 9.21
  - ridge 6.87
  - random forest 6.52
  - **xgboost 6.27 (campeón)**
- Comparación estadística: Friedman χ² = 1410 (p < 0.001), Wilcoxon con Holm, t pareada, Diebold–Mariano e IC95 % bootstrap. Ver `resultados/`.
- Registro de modelos `demanda_ia_pronostico_sku_sucursal` con versiones 1–3; alias `campeon` → versión 3.
- Evidencias: 11 capturas del panel de MLflow y 10 figuras de EDA y resultados.
- Reporte técnico en `reporte/`.
### Cambiado
- `.gitignore` excluye el binario de Random Forest (130.8 MB, supera el límite de GitHub).

## [datos-v1.1.0] — 2026-10-15 · Dataset limpio y estandarizado
### Agregado
- `fuentes/datos_prep.py` con nueve reglas de limpieza (R1–R9), reporte de calidad JSON y manifiesto con SHA-256.
- `datos/datos_limp/ventas_sku_sucursal_limpio.csv.gz`: 913,000 filas, 500 series, sin nulos. SHA-256 del contenido: `c19ee1c4…e256`.
- Columnas nuevas: `codigo_sucursal`, `codigo_sku`, `es_imputado`, `es_devolucion`, `es_anomalia`, `z_robusto`.
- `pruebas/test_datos_prep.py`: inyecta 11 errores típicos del POS y verifica que cada regla se active.
### Resultado de la limpieza
- 0 llaves inválidas, 0 duplicados exactos o de llave, 0 negativos, 0 días faltantes y 0 imputaciones.
- 1,412 anomalías marcadas (0.15 %): 1,286 picos y 126 caídas. Se conservan.

## [datos-v1.0.0] — 2026-10-15 · Datos originales
### Agregado
- Estructura del repositorio y `requirements.txt` con versiones exactas.
- `datos/datos_ini/train.csv` (Kaggle *Store Item Demand Forecasting Challenge*), sin modificaciones. SHA-256 `038f25690a65149c94f86ddd3deceda20c037a5cfd754cafdfc539a72992f2ed`.

## Próximas versiones (planeado)
- `datos-v2.0.0`: sustituir el proxy por el extracto real del ERP de Farmacias Occidente Salud, con inventario para detectar demanda censurada.
- `modelo-v1.3.0`: variables de calendario mexicano (quincenas, días festivos, temporada respiratoria) y pronóstico probabilístico (cuantiles) para el cálculo del stock de seguridad.
