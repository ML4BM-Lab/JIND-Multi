# Usa una imagen base de Python 3.7.16
FROM python:3.7.16-slim

# Establece el directorio de trabajo
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    jq \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Copia el archivo de requisitos
COPY requirements.txt .

# Copia el resto del código fuente al contenedor
COPY . .

# Instala las dependencias de Python del archivo requirements.txt
RUN pip install -e .

RUN pip install --no-cache-dir -r requirements.txt

COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh
RUN chmod 777 /app/config.json

# Exponer el puerto para la aplicación
EXPOSE 5003

# Crea un usuario no root y asigna permisos
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Define el comando por defecto para ejecutar tu aplicación
# CMD ["gunicorn", "--bind", "0.0.0.0:5003", "index:app"]
CMD ["/app/start.sh"]