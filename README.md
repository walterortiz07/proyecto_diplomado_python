# Análisis y Predicción de Volumen Diario de Llamadas

Proyecto desarrollado como parte del **Diplomado en Python**

Objetivo: Analizar métricas de un Call Center y predecir el volumen diario de llamadas mediante un modelo **SARIMA**, utilizando datos reales en intervalos de 30 minutos.

---

## Instalación y ejecución

### Clonar el repositorio
```bash
git clone https://github.com/tuusuario/proyecto_diplomado_python.git
cd proyecto_diplomado_python

### Crear entorno e instalar dependencias

python -m venv venv
source venv/bin/activate        # En Linux/Mac
venv\Scripts\activate           # En Windows

pip install -r requirements.txt

### Instalar dependencias del frontend

cd frontend
npm install
npm run build
cd ..

### Ejecutar el servidor FastAPI

python app.py


http://localhost:8000
