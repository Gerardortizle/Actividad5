"""
Project : Act 5. Baseline de pronóstico SKU–sucursal JGOL

    Name: José Gerardo Ortiz Leñero
    Matricula: AL07182099
    Date: Octubre 18 2026
    Universidad TecMilenio

**Caso de uso:** *Farmacias Occidente Salud* (cadena ficticia, 140 sucursales en el occidente de México)
train.py — Entrenamiento, ajuste (Grid Search + validación cruzada temporal),
evaluación y registro en MLflow del baseline de pronóstico de DEMANDA-IA (Fase I).

Modelos comparados (regresión, objetivo = unidades diarias por SKU–sucursal):
  0. Naive estacional (lag 7)        -> referencia mínima, sin entrenamiento
  1. Ridge (regresión lineal L2)     -> modelo lineal interpretable
  2. Random Forest                   -> ensamble por bagging, no lineal
  3. XGBoost                         -> ensamble por boosting, no lineal

Uso (desde la raíz del repositorio):
    python fuentes/train.py                    # corrida completa
    python fuentes/train.py --muestra-grid 0.05 --rapido   # prueba de humo
    mlflow ui --backend-store-uri sqlite:///mlflow.db       # abrir el panel
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.xgboost
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_prep as dp  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"
SEMILLA = 42
EXPERIMENTO = "DEMANDA-IA_FaseI_baseline_SKU_sucursal"
NOMBRE_REGISTRO = "demanda_ia_pronostico_sku_sucursal"
INICIO_TEST = "2017-07-01"                       # holdout: 2017-07-01 a 2017-12-31 (184 días)
CORTES_CV = ["2016-01-01", "2016-07-01", "2017-01-01"]  # validación: bloques de 6 meses

# ---------------------------------------------------------------------------
# Definición de modelos y rejillas (justificación en README y reporte)
# ---------------------------------------------------------------------------
CAT_RIDGE = ["sucursal_id", "sku_id", "dia_semana", "mes"]
NUM_RIDGE = [c for c in dp.FEATURES if c not in CAT_RIDGE]


def definir_modelos(rapido: bool = False) -> dict:
    ridge = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_RIDGE),
            ("num", StandardScaler(), NUM_RIDGE)])),
        ("modelo", Ridge(random_state=SEMILLA))])
    rf = RandomForestRegressor(n_estimators=60 if rapido else 100, max_samples=0.3,
                               n_jobs=-1, random_state=SEMILLA)
    xgb = XGBRegressor(n_estimators=150 if rapido else 500, subsample=0.8, colsample_bytree=0.8,
                       tree_method="hist", n_jobs=-1, random_state=SEMILLA)
    return {
        "ridge": {"estimador": ridge, "familia": "lineal",
                  "rejilla": {"modelo__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}},
        "random_forest": {"estimador": rf, "familia": "bagging",
                          "rejilla": {"max_depth": [10, 16], "min_samples_leaf": [10, 30],
                                      "max_features": [0.5, 1.0]}},
        "xgboost": {"estimador": xgb, "familia": "boosting",
                    "rejilla": {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1],
                                "objective": ["reg:squarederror", "count:poisson"]}},
    }


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def metricas(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"mae": mean_absolute_error(y, p),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": r2_score(y, p),
            "wape": float(np.abs(y - p).sum() / np.abs(y).sum()),
            "sesgo_pct": float(100 * (p.sum() - y.sum()) / y.sum())}


# ---------------------------------------------------------------------------
# Datos y particiones temporales
# ---------------------------------------------------------------------------
def cargar_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(dp.RUTA_LIMP, parse_dates=["fecha"])
    feat = dp.construir_features(df)
    train = feat[feat["fecha"] < INICIO_TEST].reset_index(drop=True)
    test = feat[feat["fecha"] >= INICIO_TEST].reset_index(drop=True)
    return train, test


def folds_temporales(fechas: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Ventana expansiva: entrena con todo lo anterior al corte, valida 6 meses posteriores.
    Evita la fuga de información que produciría un k-fold aleatorio en series de tiempo."""
    folds = []
    for corte in CORTES_CV:
        c = pd.Timestamp(corte)
        fin = c + pd.DateOffset(months=6)
        tr = np.where(fechas < c)[0]
        va = np.where((fechas >= c) & (fechas < fin))[0]
        folds.append((tr, va))
    return folds


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def graficar_modelo(nombre, test, pred, modelo, carpeta: Path) -> list[Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []
    agg = test.assign(pred=pred).groupby("fecha")[["unidades", "pred"]].sum()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(agg.index, agg["unidades"], label="Real", color="#1f3b4d", lw=1.4)
    ax.plot(agg.index, agg["pred"], label="Pronóstico", color="#e07a1f", lw=1.2)
    ax.set_title(f"{nombre}: unidades totales diarias (holdout jul–dic 2017)")
    ax.set_ylabel("Unidades"); ax.legend(); fig.tight_layout()
    r = carpeta / f"{nombre}_real_vs_pred.png"; fig.savefig(r, dpi=130); plt.close(fig); rutas.append(r)

    res = test["unidades"].values - pred
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(res, bins=80, color="#2a9d8f"); ax[0].set_title("Distribución de residuos")
    ax[0].set_xlabel("Real − pronóstico")
    ax[1].scatter(pred[::40], res[::40], s=3, alpha=.35, color="#264653")
    ax[1].axhline(0, color="red", lw=.8); ax[1].set_title("Residuo vs pronóstico (muestra 2.5%)")
    ax[1].set_xlabel("Pronóstico"); fig.tight_layout()
    r = carpeta / f"{nombre}_residuos.png"; fig.savefig(r, dpi=130); plt.close(fig); rutas.append(r)

    imp = None
    if hasattr(modelo, "feature_importances_"):
        imp = pd.Series(modelo.feature_importances_, index=dp.FEATURES)
    elif isinstance(modelo, Pipeline):
        nombres = modelo.named_steps["prep"].get_feature_names_out()
        imp = pd.Series(np.abs(modelo.named_steps["modelo"].coef_), index=nombres)
    if imp is not None:
        imp = imp.sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        imp.plot.barh(ax=ax, color="#457b9d")
        ax.set_title(f"{nombre}: importancia de variables" + (" (|coef| estandarizado)" if isinstance(modelo, Pipeline) else ""))
        fig.tight_layout()
        r = carpeta / f"{nombre}_importancia.png"; fig.savefig(r, dpi=130); plt.close(fig); rutas.append(r)
    return rutas


# ---------------------------------------------------------------------------
# Comparación estadística
# ---------------------------------------------------------------------------
def diebold_mariano(e1, e2, fechas, h: int = dp.HORIZONTE) -> dict:
    """DM sobre la pérdida absoluta media diaria; varianza HAC (Newey-West, h-1 rezagos)
    y corrección de Harvey, Leybourne y Newbold (1997)."""
    d = pd.Series(np.abs(e1) - np.abs(e2)).groupby(np.asarray(fechas)).mean().values
    T, dm = len(d), d.mean()
    gamma = [np.mean((d[k:] - dm) * (d[:T - k] - dm)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    stat = dm / np.sqrt(var)
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return {"dm_estadistico": float(stat), "p_valor": float(p), "dias": int(T)}


def holm(pvals: dict) -> dict:
    orden = sorted(pvals.items(), key=lambda kv: kv[1])
    m, ajust, previo = len(orden), {}, 0.0
    for i, (k, p) in enumerate(orden):
        previo = max(previo, min(1.0, (m - i) * p))
        ajust[k] = previo
    return ajust


def comparar(test: pd.DataFrame, preds: dict, cv_maes: dict) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(SEMILLA)
    errores = {m: test["unidades"].values - p for m, p in preds.items()}
    mae_serie = pd.DataFrame({m: pd.Series(np.abs(e)).groupby([test["sucursal_id"], test["sku_id"]]).mean().values
                              for m, e in errores.items()})
    modelos = list(preds)
    fr = stats.friedmanchisquare(*[mae_serie[m] for m in modelos])
    rangos = mae_serie.rank(axis=1).mean().to_dict()
    pares, p_wil = [], {}
    for i, a in enumerate(modelos):
        for b in modelos[i + 1:]:
            dif = mae_serie[a] - mae_serie[b]
            w = stats.wilcoxon(mae_serie[a], mae_serie[b])
            t = stats.ttest_rel(mae_serie[a], mae_serie[b])
            boot = [dif.sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
            dm = diebold_mariano(errores[a], errores[b], test["fecha"])
            clave = f"{a} vs {b}"
            p_wil[clave] = w.pvalue
            pares.append({"comparacion": clave, "dif_mae_media": dif.mean(),
                          "ic95_inf": np.percentile(boot, 2.5), "ic95_sup": np.percentile(boot, 97.5),
                          "series_donde_A_gana": int((dif < 0).sum()),
                          "wilcoxon_p": w.pvalue, "t_pareada_p": t.pvalue,
                          "efecto_d_cohen": dif.mean() / dif.std(ddof=1),
                          "dm_estadistico": dm["dm_estadistico"], "dm_p": dm["p_valor"]})
    tabla = pd.DataFrame(pares)
    aj = holm(p_wil)
    tabla["wilcoxon_p_holm"] = tabla["comparacion"].map(aj)
    resumen = {"friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue),
               "rango_promedio_por_modelo": rangos,
               "mae_cv_por_fold": {m: list(map(float, v)) for m, v in cv_maes.items()},
               "n_series": int(len(mae_serie))}
    return resumen, tabla


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------
def commit_git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, text=True).strip()
    except Exception:
        return "sin_git"


def main(args) -> None:
    np.random.seed(SEMILLA)
    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.set_experiment(EXPERIMENTO)
    client = MlflowClient()
    client.set_experiment_tag(exp.experiment_id, "mlflow.note.content",
        "Baseline Fase I de DEMANDA-IA (Farmacias Occidente Salud): pronóstico diario de unidades por SKU–sucursal "
        "con horizonte de 7 días. Comparación Naive estacional vs Ridge vs Random Forest vs XGBoost.")
    for k, v in {"proyecto": "DEMANDA-IA", "fase": "I", "cliente": "Farmacias Occidente Salud",
                 "tarea": "regresion_series_tiempo"}.items():
        client.set_experiment_tag(exp.experiment_id, k, v)

    t0 = time.time()
    train, test = cargar_features()
    rep = json.loads(dp.RUTA_REPORTE.read_text(encoding="utf-8"))
    print(f"train={len(train):,} filas  test={len(test):,} filas  ({time.time()-t0:.0f}s)")

    muestra = train.sample(frac=args.muestra_grid, random_state=SEMILLA).sort_values("fecha").reset_index(drop=True)
    cv = folds_temporales(muestra["fecha"])
    dataset_ml = mlflow.data.from_pandas(
        train[["fecha"] + dp.FEATURES + [dp.OBJETIVO]].head(5000), name=f"ventas_sku_sucursal_v{rep['version_datos']}",
        targets=dp.OBJETIVO, digest=rep["sha256_contenido_limpio"][:8])

    comunes = {"semilla": SEMILLA, "horizonte_dias": dp.HORIZONTE, "inicio_test": INICIO_TEST,
               "cortes_cv": ",".join(CORTES_CV), "esquema_cv": "ventana_expansiva_3_folds_6m",
               "muestra_grid_frac": args.muestra_grid, "filas_grid": len(muestra),
               "filas_train": len(train), "filas_test": len(test), "n_features": len(dp.FEATURES),
               "version_datos": rep["version_datos"]}
    etiquetas = {"git_commit": commit_git(), "dataset_sha256": rep["sha256_contenido_limpio"],
                 "python": platform.python_version(), "sklearn": sklearn.__version__,
                 "xgboost": xgboost.__version__, "mlflow": mlflow.__version__, "autor": "J. Gerardo Ortiz Leñero"}

    preds, cv_maes, resumen_metricas = {}, {}, []
    carpeta_graf = RESULTADOS / "graficas"

    # --- 0. Baseline naive estacional --------------------------------------
    with mlflow.start_run(run_name="00_naive_estacional_lag7") as run:
        mlflow.set_tags({**etiquetas, "modelo": "naive_estacional", "familia": "referencia", "rol": "baseline"})
        mlflow.log_params({**comunes, "regla": "y_hat(t) = y(t-7)"})
        p = test["lag_7"].values
        m = metricas(test["unidades"], p)
        mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
        cv_maes["naive_estacional"] = [mean_absolute_error(muestra.loc[va, "unidades"], muestra.loc[va, "lag_7"]) for _, va in cv]
        for i, v in enumerate(cv_maes["naive_estacional"]):
            mlflow.log_metric("cv_mae_fold", v, step=i + 1)
        for r in graficar_modelo("naive_estacional", test, p, None, carpeta_graf):
            mlflow.log_artifact(str(r), "graficas")
        preds["naive_estacional"] = p
        resumen_metricas.append({"modelo": "naive_estacional", **m, "cv_mae_media": np.mean(cv_maes["naive_estacional"]),
                                 "cv_mae_std": np.std(cv_maes["naive_estacional"]), "seg_entrenamiento": 0.0,
                                 "mejores_hiperparametros": "n/a", "run_id": run.info.run_id})

    # --- 1-3. Modelos con Grid Search --------------------------------------
    for orden, (nombre, cfg) in enumerate(definir_modelos(args.rapido).items(), start=1):
        print(f"\n== {nombre}: grid de {np.prod([len(v) for v in cfg['rejilla'].values()])} combinaciones x {len(cv)} folds")
        with mlflow.start_run(run_name=f"0{orden}_{nombre}") as padre:
            mlflow.set_tags({**etiquetas, "modelo": nombre, "familia": cfg["familia"], "rol": "candidato"})
            mlflow.log_params(comunes)
            mlflow.log_input(dataset_ml, context="entrenamiento")
            gs = GridSearchCV(cfg["estimador"], cfg["rejilla"], cv=cv, n_jobs=1, refit=False,
                              scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"})
            t1 = time.time()
            gs.fit(muestra[dp.FEATURES], muestra[dp.OBJETIVO])
            seg_grid = time.time() - t1
            cvr = pd.DataFrame(gs.cv_results_)

            # Corridas hijas: una por combinación de hiperparámetros
            for _, fila in cvr.iterrows():
                etiqueta = ",".join(f"{k.split('__')[-1]}={v}" for k, v in fila["params"].items())
                with mlflow.start_run(run_name=f"{nombre}|{etiqueta}", nested=True):
                    mlflow.set_tags({"modelo": nombre, "rol": "combinacion_grid", "rank_mae": int(fila["rank_test_mae"])})
                    mlflow.log_params({k.split("__")[-1]: v for k, v in fila["params"].items()})
                    mlflow.log_metrics({"cv_mae_media": -fila["mean_test_mae"], "cv_mae_std": fila["std_test_mae"],
                                        "cv_rmse_media": -fila["mean_test_rmse"], "cv_r2_media": fila["mean_test_r2"],
                                        "seg_ajuste_promedio": fila["mean_fit_time"]})

            mejor = cvr.loc[cvr["rank_test_mae"].idxmin()]
            mejores = mejor["params"]
            cv_maes[nombre] = [-mejor[f"split{i}_test_mae"] for i in range(len(cv))]
            for i, v in enumerate(cv_maes[nombre]):
                mlflow.log_metric("cv_mae_fold", v, step=i + 1)
            mlflow.log_params({f"mejor_{k.split('__')[-1]}": v for k, v in mejores.items()})
            mlflow.log_params({f"fijo_{k}": v for k, v in (cfg["estimador"].get_params().items()
                               if nombre != "ridge" else []) if k in ("n_estimators", "subsample", "colsample_bytree", "max_samples", "tree_method")})
            mlflow.log_metrics({"cv_mae_media": -mejor["mean_test_mae"], "cv_mae_std": mejor["std_test_mae"],
                                "cv_rmse_media": -mejor["mean_test_rmse"], "cv_r2_media": mejor["mean_test_r2"],
                                "seg_grid_search": seg_grid})

            # Reentrenamiento con todo el periodo de entrenamiento y evaluación en holdout
            modelo = sklearn.base.clone(cfg["estimador"]).set_params(**mejores)
            t1 = time.time()
            modelo.fit(train[dp.FEATURES], train[dp.OBJETIVO])
            seg_fit = time.time() - t1
            p = np.clip(modelo.predict(test[dp.FEATURES]), 0, None)
            m = metricas(test["unidades"], p)
            mlflow.log_metrics({**{f"test_{k}": v for k, v in m.items()}, "seg_entrenamiento_final": seg_fit})
            print(f"   mejor={mejores}  cv_mae={-mejor['mean_test_mae']:.3f}  test_mae={m['mae']:.3f}  ({seg_grid+seg_fit:.0f}s)")

            ruta_cv = RESULTADOS / f"cv_resultados_{nombre}.csv"
            RESULTADOS.mkdir(exist_ok=True)
            cvr.drop(columns=["params"]).to_csv(ruta_cv, index=False)
            mlflow.log_artifact(str(ruta_cv), "grid_search")
            for r in graficar_modelo(nombre, test, p, modelo, carpeta_graf):
                mlflow.log_artifact(str(r), "graficas")
            mlflow.log_artifact(str(dp.RUTA_REPORTE), "datos")
            mlflow.log_artifact(str(RAIZ / "requirements.txt"), "entorno")
            firma = infer_signature(test[dp.FEATURES].head(200), p[:200])
            kw = dict(name="modelo", signature=firma, input_example=test[dp.FEATURES].head(3),
                      registered_model_name=NOMBRE_REGISTRO)
            if nombre == "xgboost":
                info = mlflow.xgboost.log_model(modelo, **kw)
            else:   # skops: se declara explícitamente el único tipo confiable que usa el bosque
                info = mlflow.sklearn.log_model(modelo, skops_trusted_types=["sklearn.tree._tree.Tree"]
                                                if nombre == "random_forest" else None, **kw)
            preds[nombre] = p
            resumen_metricas.append({"modelo": nombre, **m, "cv_mae_media": -mejor["mean_test_mae"],
                                     "cv_mae_std": mejor["std_test_mae"], "seg_entrenamiento": seg_fit,
                                     "mejores_hiperparametros": json.dumps(mejores),
                                     "run_id": padre.info.run_id, "version_registro": info.registered_model_version})

    # --- Comparación estadística y selección del campeón -------------------
    tabla_m = pd.DataFrame(resumen_metricas)
    resumen, tabla_p = comparar(test, preds, cv_maes)
    campeon = tabla_m.loc[tabla_m["mae"].idxmin()]
    tabla_m.to_csv(RESULTADOS / "tabla_metricas.csv", index=False)
    tabla_p.to_csv(RESULTADOS / "comparacion_pares.csv", index=False)
    resumen["campeon"] = campeon["modelo"]
    (RESULTADOS / "comparacion_estadistica.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"fecha": test["fecha"], "codigo_sucursal": "SUC-" + test["sucursal_id"].astype(str).str.zfill(2),
                  "codigo_sku": "SKU-" + test["sku_id"].astype(str).str.zfill(3), "real": test["unidades"],
                  **{f"pred_{k}": np.round(v, 3) for k, v in preds.items()}}
                 ).to_csv(RESULTADOS / "predicciones_test.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

    with mlflow.start_run(run_name="99_comparacion_estadistica"):
        mlflow.set_tags({**etiquetas, "rol": "comparacion", "campeon": campeon["modelo"]})
        mlflow.log_params({"pruebas": "Friedman, Wilcoxon(Holm), t pareada, Diebold-Mariano(HLN), bootstrap IC95",
                           "unidad_comparacion": "MAE por serie SKU-sucursal (n=500)"})
        mlflow.log_metrics({"friedman_chi2": resumen["friedman_chi2"], "friedman_p": resumen["friedman_p"]})
        for f in ["tabla_metricas.csv", "comparacion_pares.csv", "comparacion_estadistica.json"]:
            mlflow.log_artifact(str(RESULTADOS / f), "comparacion")

    if "version_registro" in campeon and pd.notna(campeon.get("version_registro")):
        client.set_registered_model_alias(NOMBRE_REGISTRO, "campeon", str(int(campeon["version_registro"])))
        client.update_registered_model(NOMBRE_REGISTRO, description=(
            "Pronóstico diario de unidades por SKU–sucursal (horizonte 7 días) para DEMANDA-IA. "
            "El alias 'campeon' apunta a la versión con menor MAE en el holdout jul–dic 2017."))
        for _, f in tabla_m.dropna(subset=["version_registro"]).iterrows():
            client.set_model_version_tag(NOMBRE_REGISTRO, str(int(f["version_registro"])), "algoritmo", f["modelo"])
    print("\n", tabla_m[["modelo", "mae", "rmse", "r2", "wape", "cv_mae_media"]].round(4).to_string(index=False))
    print(f"\nCampeón: {campeon['modelo']}  | tiempo total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrenamiento DEMANDA-IA con MLflow")
    ap.add_argument("--tracking-uri", default=f"sqlite:///{RAIZ / 'mlflow.db'}")
    ap.add_argument("--muestra-grid", type=float, default=0.12,
                    help="Fracción del entrenamiento usada en Grid Search (costo computacional)")
    ap.add_argument("--rapido", action="store_true", help="Menos árboles; sólo para pruebas de humo")
    main(ap.parse_args())
