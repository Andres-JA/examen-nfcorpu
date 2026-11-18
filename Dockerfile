FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nfcorpus_ir.py app.py ./
COPY datos/nfcorpus/corpus ./datos/nfcorpus/corpus
ENV PORT=8000
EXPOSE 8000
CMD ["python", "-c", "import os; from waitress import serve; from app import create_app; serve(create_app(), host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))"]
