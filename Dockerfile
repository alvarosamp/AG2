FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py app.py ./
COPY dados ./dados

# Treina o modelo em build time: a imagem já sobe pronta para prever.
RUN python main.py --sem-interacao

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
