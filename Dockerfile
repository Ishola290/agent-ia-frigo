FROM python:3.11-slim

WORKDIR /app

# Copier les fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p models datasets logs

# Exposer le port
EXPOSE 5000

# Lancer l'application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "agent_ia:app"]