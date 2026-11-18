# Resolución NFCorpus

Abrir `resolucion_nfcorpus.ipynb`. Contiene el código, las salidas ejecutadas,
la evaluación de las 323 consultas test y la comparación TF-IDF/BM25.
El enunciado original `examen1bim.ipynb` se conserva sin modificar.

Instalación: `python -m pip install -r requirements.txt`.
Servicio local: `python app.py`, en `http://127.0.0.1:8000`.
El notebook documenta las solicitudes de Postman y el despliegue con Docker.

Búsqueda exclusivamente por `POST /search`, según el enunciado.
En Postman usar Body → raw → JSON y Content-Type: application/json:
`{"query":"diet and cancer","model":"bm25","top_k":10}`.
La disponibilidad se comprueba con `GET /health`. GET /search devuelve HTTP 405.
Para actualizar Render, reemplazar app.py en GitHub y desplegar el nuevo commit.

Los datos originales y sus atribuciones están en `datos/`.
Las métricas, rankings y consultas evaluadas están en `resultados/`.
La disponibilidad de una URL pública se indica en la última celda del notebook.
