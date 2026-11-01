# ProteinovaLabs · AI Media Optimizer (MVP)

## Ejecutar localmente
1. Instalar Python 3.10 o superior.
2. Abrir una terminal en esta carpeta.
3. Ejecutar:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Se abrirá la aplicación en el navegador.

## Publicar para obtener una URL y crear el QR
1. Crear un repositorio en GitHub y subir `app.py` y `requirements.txt` (también este README).
2. Entrar en https://share.streamlit.io/ e iniciar sesión con GitHub.
3. Crear una app nueva, elegir el repositorio, rama y archivo `app.py`.
4. Desplegar. Streamlit generará una URL pública.
5. Probar la URL en una ventana privada/móvil y recién entonces generar el QR para la diapositiva.

## Alcance y límites
La app reproduce el flujo del notebook: muestreo Sobol, ensemble MLP, algoritmo genético, proxies de pH/osmolalidad, ranking y CSV. Para que el entrenamiento sea más estable con variables de escalas muy distintas, el MLP se implementa dentro de un pipeline con `StandardScaler`.

**Advertencia:** `simulated_growth()` es una función matemática sintética de demostración. No es un modelo biológico validado ni está entrenado con datos experimentales. Los costos son índices ilustrativos derivados de los coeficientes del notebook; no deben interpretarse como USD/L reales. Los valores de pH/osmolalidad son proxies. No usar los resultados para formular medios reales.
