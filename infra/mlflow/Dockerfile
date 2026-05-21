FROM ghcr.io/mlflow/mlflow:latest

# Drivers requis par MLflow tracking server :
#   - psycopg2 : pour le backend Postgres (metadata des runs)
#   - boto3    : pour l'artifact store S3-compatible (MinIO ici)
RUN pip install --no-cache-dir psycopg2-binary boto3
