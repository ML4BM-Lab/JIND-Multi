# Usa una imagen base de Python 3.7.16
FROM python:3.7.16-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requisitos
COPY requirements.txt .

# Instala las dependencias de Python del archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código fuente al contenedor
COPY . .

# Exponer el puerto para la aplicación
EXPOSE 5003

# Crea un usuario no root y asigna permisos
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Define el comando por defecto para ejecutar tu aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:5003", "index:app"]