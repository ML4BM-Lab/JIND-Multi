# Usa una imagen base de Ubuntu
FROM ubuntu:20.04

# Establece variables de entorno para evitar preguntas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.7.16

# Instala las dependencias necesarias para compilar Python
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    wget \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Descarga y compila Python desde el código fuente
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-$PYTHON_VERSION Python-$PYTHON_VERSION.tgz

# Establece el directorio de trabajo
WORKDIR /app

# Copia el contenido de tu proyecto al contenedor
COPY . /app/

# Instala pip y las dependencias
RUN python3.7 -m ensurepip && \
    python3.7 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Instala las dependencias de tu proyecto
RUN python3.7 -m pip install --no-cache-dir -e .

RUN pip3.7 install --no-cache-dir -r requirements.txt

EXPOSE 5003

# Define el comando por defecto para ejecutar tu aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:5003", "index:app"]