import { useEffect, useRef, useState } from "react";
import { Chart } from "chart.js/auto";

export default function App() {
  const [data, setData] = useState(null);
  const mainChartRef = useRef(null);
  const compareChartRef = useRef(null);

  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_BASE || "http://localhost:8000"}/api/analizar`)
      .then((res) => res.json())
      .then(setData)
      .catch((err) => console.error("Error al cargar datos:", err));
  }, []);


  useEffect(() => {
    if (!data) return;
    drawMainChart();
    drawCompareChart();
  }, [data]);


  const drawMainChart = () => {
    const ctx = mainChartRef.current.getContext("2d");

    const allDates = data.series.map((x) => x.fecha);
    const allValues = data.series.map((x) => x.llamadas_totales);

    const predDates = data.predictions.map((x) => x.fecha);
    const predValues = data.predictions.map((x) => x.pred);

    const validDates = data.validation.fechas;
    const validValues = data.validation.real;

    new Chart(ctx, {
      type: "line",
      data: {
        labels: allDates,
        datasets: [
          {
            label: "Entrenamiento (hasta agosto)",
            data: allValues,
            borderColor: "steelblue",
            borderWidth: 2,
            tension: 0.3,
          },
          {
            label: "Predicción SARIMA",
            data: predValues,
            borderColor: "orange",
            borderDash: [5, 5],
            borderWidth: 2,
            tension: 0.3,
          },
          {
            label: "Real (octubre)",
            data: validValues,
            borderColor: "green",
            borderWidth: 2,
            tension: 0.3,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "top" },
          title: {
            display: true,
            text: `SARIMA - Entrenamiento hasta ${data.fechas_clave.train_hasta}, validación desde ${data.fechas_clave.valid_desde}`,
          },
        },
        scales: {
          x: { ticks: { maxTicksLimit: 10 } },
          y: { beginAtZero: true },
        },
      },
    });
  };

  // --- Gráfico comparativo
  const drawCompareChart = () => {
    const ctx = compareChartRef.current.getContext("2d");

    new Chart(ctx, {
      type: "line",
      data: {
        labels: data.validation.fechas,
        datasets: [
          {
            label: "Real (octubre)",
            data: data.validation.real,
            borderColor: "green",
            borderWidth: 2,
          },
          {
            label: "Predicción (octubre)",
            data: data.validation.pred,
            borderColor: "orange",
            borderDash: [5, 5],
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "top" },
          title: {
            display: true,
            text: "Comparación octubre 2025: Predicción vs Real",
          },
        },
        scales: {
          y: { beginAtZero: true },
        },
      },
    });
  };

  if (!data) return <p style={{ padding: 20 }}>Cargando datos desde llamadas_callcenter.csv...</p>;

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "20px" }}>
      <h1>Predicción SARIMA</h1>
      <p>
        Entrenamiento hasta <strong>{data.fechas_clave.train_hasta}</strong> — Validación desde{" "}
        <strong>{data.fechas_clave.valid_desde}</strong>
      </p>

      <div style={{ background: "#fff", padding: "20px", borderRadius: "12px", marginBottom: "24px" }}>
        <canvas ref={mainChartRef} height="120"></canvas>
      </div>

      <div style={{ background: "#fff", padding: "20px", borderRadius: "12px", marginBottom: "24px" }}>
        <canvas ref={compareChartRef} height="120"></canvas>
      </div>

      <div
        style={{
          background: "#f4f4f4",
          padding: "16px 24px",
          borderRadius: "12px",
          lineHeight: "1.6em",
        }}
      >
        <h3>Métricas del modelo (Octubre 2025)</h3>
        <p>
          <strong>RMSE:</strong> {data.validation.rmse.toFixed(2)} | <strong>MAE:</strong>{" "}
          {data.validation.mae.toFixed(2)} | <strong>R²:</strong> {data.validation.r2.toFixed(3)}
        </p>
        <p>
          <strong>SLA promedio:</strong> {data.stats.promedio_sla.toFixed(2)}% |{" "}
          <strong>Abandono promedio:</strong> {data.stats.promedio_abandono.toFixed(2)}%
        </p>
        <p>
          <strong>Días pico:</strong> {data.stats.dias_pico.join(", ") || "Ninguno"}
          <br />
          <strong>Días valle:</strong> {data.stats.dias_valle.join(", ") || "Ninguno"}
        </p>
      </div>
    </div>
  );
}
