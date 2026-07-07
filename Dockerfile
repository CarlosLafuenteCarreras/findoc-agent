
ccat > .dockerignore
ookworm

WORKDIR /app

# Dependencias del sistema (necesarias para ChromaDB y PyMuPDF)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/

# Los datos (chroma_db, PDFs, dataset) NO se copian aquí
# — se añaden solo en el repo del Space, no en el de GitHub

# Puerto que usa Streamlit en HuggingFace
EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]