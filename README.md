# Actividad 5 · DEMANDA-IA — Baseline de pronóstico SKU–sucursal

**Proyecto:** DEMANDA-IA, Fase I · **Cliente (caso):** Farmacias Occidente Salud
**Curso:** Gestión de proyectos de inteligencia artificial — Universidad Tecmilenio
**Autor:** José Gerardo Ortiz Leñero

Primer baseline del componente central de la Fase I: un dataset temporal a nivel SKU–sucursal, limpio y versionado, y la comparación de modelos de pronóstico de demanda diaria entrenados con Grid Search, validación cruzada temporal y registro completo en MLflow.

## Resultado en una línea

XGBoost es el modelo campeón: **MAE 6.27 unidades, WAPE 10.2 %, R² 0.936** en el holdout jul–dic 2017. Reduce el error 32 % frente a la referencia naive estacional y es significativamente mejor que Ridge y Random Forest (Friedman p < 0.001; Wilcoxon-Holm p < 0.001; Diebold–Mariano p < 0.001).

| Modelo | MAE | RMSE | R² | WAPE | MAE CV (media ± d.e.) |
|---|---|---|---|---|---|
| Naive estacional (lag 7) | 9.21 | 12.11 | 0.859 | 15.0 % | 8.60 ± 0.32 |
| Ridge (α = 100) | 6.87 | 8.87 | 0.924 | 11.2 % | 6.46 ± 0.25 |
| Random Forest | 6.52 | 8.46 | 0.931 | 10.6 % | 6.44 ± 0.22 |
| **XGBoost** | **6.27** | **8.13** | **0.936** | **10.2 %** | **6.18 ± 0.17** |

## Estructura del repositorio

```
Actividad5/
├── datos/
│   ├── datos_ini/train.csv                       # datos originales, sin modificar (v1.0.0)
│   ├── datos_limp/ventas_sku_sucursal_limpio.csv.gz  # datos limpios (v1.1.0)
│   ├── datos_limp/reporte_calidad.json           # conteo por regla de limpieza
│   └── MANIFIESTO_DATOS.json                     # versión + SHA-256 de cada archivo de datos
├── fuentes/
│   ├── entrena.ipynb    # cuaderno de Colab: EDA, entrenamiento, evaluación y trazabilidad
│   ├── datos_prep.py    # funciones de limpieza (reglas R1–R9) y de ingeniería de variables
│   └── train.py         # Grid Search + validación temporal + registro en MLflow
├── pruebas/test_datos_prep.py   # prueba de robustez de la limpieza con datos contaminados
├── resultados/          # métricas, comparación estadística, predicciones y gráficas por modelo
├── evidencias/
│   ├── graficas/        # figuras del EDA y de resultados usadas en el reporte
│   └── mlflow/          # capturas del panel de MLflow
├── mlflow.db            # backend de seguimiento (SQLite): corridas, parámetros, métricas, registro
├── mlruns/              # artefactos de MLflow (gráficas, CSV, modelos)
├── reporte/             # reporte técnico (Word y PDF)
├── requirements.txt     # versiones exactas
├── README.md
└── CHANGELOG.md
```

## Cómo reproducir

### Opción A · Google Colab
1. Abre `fuentes/entrena.ipynb` en Colab.
2. En la primera celda, sustituye `URL_REPO` por la URL de tu repositorio.
3. Ejecuta todo. Si pip actualiza pandas, reinicia el entorno y vuelve a ejecutar.

### Opción B · Local
```bash
git clone <URL_REPO> && cd Actividad5
python -m venv .venv && source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python fuentes/datos_prep.py          # regenera datos_limp y el reporte de calidad
python pruebas/test_datos_prep.py     # verifica las reglas de limpieza
python fuentes/train.py               # ~7 min con 1 CPU; crea corridas en mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db       # http://127.0.0.1:5000
```
Todos los comandos se ejecutan **desde la raíz del repositorio**, porque MLflow guarda los artefactos en `./mlruns`.

La semilla (`42`), los cortes temporales y las rejillas están fijos en `train.py`. Además, cada corrida registra en MLflow el commit de Git, el SHA-256 del dataset y las versiones de Python y de las librerías.

## Datos

- **Fuente:** Kaggle, *Store Item Demand Forecasting Challenge*, archivo `train.csv`. Contiene 913,000 filas de ventas diarias de 50 artículos en 10 tiendas, del 01-01-2013 al 31-12-2017. La copia se descargó de un espejo público en GitHub; su SHA-256 es `038f2569…92f2ed`.
- **Por qué este dataset:** tiene exactamente la granularidad que requiere DEMANDA-IA (fecha × sucursal × SKU, unidades diarias). Funciona como **proxy estructural** mientras se obtiene el extracto real del ERP de Farmacias Occidente Salud.
- **Supuestos:** `store` equivale a sucursal y `item` a SKU. `sales` son unidades dispensadas, no la demanda censurada por desabasto. No hay precios, promociones ni inventario.

### Diccionario de datos (dataset limpio)

| Columna | Tipo | Descripción |
|---|---|---|
| fecha | fecha | Día de venta |
| sucursal_id / codigo_sucursal | int / texto | 1–10 / `SUC-01` … `SUC-10` |
| sku_id / codigo_sku | int / texto | 1–50 / `SKU-001` … `SKU-050` |
| unidades | int | Unidades vendidas (objetivo) |
| es_imputado | 0/1 | Valor imputado por la regla R7 |
| es_devolucion | 0/1 | Valor original negativo (R5) |
| es_anomalia | 0/1 | \|z robusto\| > 3.5 (R8), se conserva |
| z_robusto | float | Desviación robusta frente a la mediana móvil de 28 días |

## Decisiones técnicas

| Decisión | Justificación |
|---|---|
| Horizonte de 7 días (todas las variables usan rezagos ≥ 7) | El pronóstico nocturno debe cubrir el tiempo de reabasto. Además, evita la fuga de información. |
| Validación de ventana expansiva, 3 folds de 6 meses | Un k-fold aleatorio mezclaría futuro y pasado. Los bloques de 6 meses abarcan ambas mitades estacionales del año. |
| Holdout jul–dic 2017, que no se toca durante el ajuste | Estimación honesta del error de producción. |
| Grid Search sobre una muestra del 12 % y reentrenamiento con el 100 % | El costo computacional baja ~8× (1 CPU). El ranking de hiperparámetros es estable con muestras de este tamaño. |
| Métrica de selección: MAE, con WAPE como métrica de negocio | El MAE se expresa en unidades y es robusto a picos. El WAPE es comparable entre SKU de distinto volumen. El RMSE y el R² se reportan como complemento. |
| Anomalías marcadas, no eliminadas | Un pico real (p. ej., temporada respiratoria) es la señal que el modelo debe aprender. |
| Ridge con one-hot y escalado; árboles sin escalar | La regularización L2 requiere variables en la misma escala. Los árboles son invariantes a la escala. |

Las rejillas de hiperparámetros, con su justificación técnica, están documentadas en el reporte (sección 6).

## Versionado

- **Código y datos:** Git, con etiquetas semánticas. `datos-v1.0.0` corresponde al original; `datos-v1.1.0`, al limpio; `modelo-v1.2.0`, al baseline entrenado. Ver `CHANGELOG.md`.
- **Integridad:** `datos/MANIFIESTO_DATOS.json` guarda el SHA-256 de cada versión. `train.py` registra el hash del contenido limpio como tag de cada corrida.
- **Modelos:** registro de MLflow `demanda_ia_pronostico_sku_sucursal`, con 3 versiones (ridge, random_forest y xgboost). El alias `campeon` apunta a la versión 3 (XGBoost). El proceso nocturno cargaría `models:/demanda_ia_pronostico_sku_sucursal@campeon`.

> **Nota:** el binario del Random Forest (~130 MB) supera el límite de 100 MB de GitHub, por eso está excluido en `.gitignore`. Se regenera con `train.py`. Esto aparece registrado como deuda técnica en el reporte.

## Evidencia de MLflow

El experimento `DEMANDA-IA_FaseI_baseline_SKU_sucursal` tiene 30 corridas:
- 1 corrida de referencia (naive).
- 3 corridas padre, una por algoritmo, con sus mejores hiperparámetros, el MAE por fold, las métricas de test, gráficas, resultados del grid y el modelo con firma y ejemplo de entrada.
- 25 corridas hijas, una por combinación del grid.
- 1 corrida con la comparación estadística.

Las capturas están en `evidencias/mlflow/`.

## Publicar en GitHub

```bash
git remote add origin https://github.com/<tu_usuario>/Actividad5.git
git push -u origin main --tags      # --tags publica datos-v1.0.0, datos-v1.1.0 y modelo-v1.2.0
```
Antes del primer commit propio, configura tu identidad con `git config user.email "<tu correo de GitHub>"`.
