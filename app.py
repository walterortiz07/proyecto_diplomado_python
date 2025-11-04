from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


# configuracion de fastAPI
app = FastAPI(title="SARIMA CallCenter Temporal Split", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# path del csv
CSV_PATH = os.path.join(os.path.dirname(__file__), "llamadas_callcenter.csv")

# usamos la fecha de intervalo y separamos en fecha y hora
def preparar_fecha(df):
    """Limpia la columna inicio_del_intervalo y crea fecha/hora separadas."""
    s = df["inicio_del_intervalo"].astype(str).str.strip()
    split_df = s.str.split(" ", n=1, expand=True)
    split_df.columns = ["fecha_str", "hora_str"]
    df["fecha"] = pd.to_datetime(split_df["fecha_str"], dayfirst=True, errors="coerce").dt.date
    df["hora"] = split_df["hora_str"].fillna("00:00")
    cols = ["fecha", "hora"] + [c for c in df.columns if c not in ["fecha", "hora", "inicio_del_intervalo"]]
    return df[cols]


def agrupar_diario(df):
    """Agrupa por fecha sumando llamadas y calcula métricas complementarias."""
    for col in ["contestadas", "abandonadas", "cumplen_el_sla"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    diario = (
        df.groupby("fecha")
        .agg(
            llamadas_totales=("contestadas", "sum"),
            abandonadas=("abandonadas", "sum"),
            sla_cumplido=("cumplen_el_sla", "sum"),
        )
        .reset_index()
    )

    diario["fecha"] = pd.to_datetime(diario["fecha"])
    diario["tasa_abandono"] = (
        (diario["abandonadas"] / (diario["llamadas_totales"] + diario["abandonadas"]).replace(0, np.nan)) * 100
    )
    diario["tasa_sla"] = (
        (diario["sla_cumplido"] / diario["llamadas_totales"].replace(0, np.nan)) * 100
    )
    return diario


# a partir de aca los endpoints de la API
@app.get("/api/analizar")
def analizar():
    """Lee el CSV, entrena con datos hasta agosto y compara con octubre."""
    if not os.path.exists(CSV_PATH):
        return {"error": f"No se encontró el archivo {CSV_PATH}"}

    df = pd.read_csv(CSV_PATH)
    df = preparar_fecha(df)
    diario = agrupar_diario(df)

    # Serie completa (diaria)
    serie = diario.set_index("fecha")["llamadas_totales"].astype(float).fillna(0)
    serie = serie[serie.index.notna()]

    # División temporal
    fecha_entrenamiento = "2025-08-31"
    fecha_validacion = "2025-10-01"

    serie_entrenamiento = serie.loc[:fecha_entrenamiento]
    serie_validacion_real = serie.loc[fecha_validacion:]

    if serie_entrenamiento.empty or serie_validacion_real.empty:
        return {"error": "Los datos no contienen registros suficientes para las fechas definidas."}

    # Entrenamiento SARIMA
    model = SARIMAX(serie_entrenamiento, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    res = model.fit(disp=False)

    # Predicción desde septiembre hasta el fin de octubre
    fecha_inicio_pred = pd.to_datetime("2025-09-01")
    fecha_fin_pred = serie_validacion_real.index.max()
    dias_pred = (fecha_fin_pred - fecha_inicio_pred).days + 1
    pred_idx = pd.date_range(fecha_inicio_pred, periods=dias_pred, freq="D")

    preds = res.predict(start=len(serie_entrenamiento), end=len(serie_entrenamiento) + dias_pred - 1)
    preds.index = pred_idx

    # Recorte a octubre (comparación real vs predicción)
    pred_oct = preds.loc["2025-10-01":"2025-10-31"]
    real_oct = serie_validacion_real.loc["2025-10-01":"2025-10-31"]

    # Métricas de octubre
    rmse = float(np.sqrt(mean_squared_error(real_oct, pred_oct)))
    mae = float(mean_absolute_error(real_oct, pred_oct))
    r2 = float(r2_score(real_oct, pred_oct))

    # Estadísticas generales
    mu = serie.mean()
    sd = serie.std()
    peaks = serie[serie > mu + 2 * sd]
    valleys = serie[serie < mu - 2 * sd]

    # Respuesta JSON
    return {
        "series": [{"fecha": str(k.date()), "llamadas_totales": float(v)} for k, v in serie.items()],
        "predictions": [{"fecha": str(k.date()), "pred": float(v)} for k, v in preds.items()],
        "validation": {
            "fechas": [str(d) for d in real_oct.index],
            "real": real_oct.tolist(),
            "pred": pred_oct.tolist(),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        },
        "stats": {
            "media": mu,
            "desviacion": sd,
            "dias_pico": peaks.index.strftime("%Y-%m-%d").tolist(),
            "dias_valle": valleys.index.strftime("%Y-%m-%d").tolist(),
            "promedio_sla": float(diario["tasa_sla"].mean()),
            "promedio_abandono": float(diario["tasa_abandono"].mean()),
        },
        "fechas_clave": {
            "train_hasta": fecha_entrenamiento,
            "valid_desde": fecha_validacion,
        },
    }


# path del frontend
frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

    @app.get("/{path_name:path}")
    def serve_frontend(path_name: str):
        """Devuelve el build de Vite si existe."""
        return FileResponse(os.path.join(frontend_path, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
