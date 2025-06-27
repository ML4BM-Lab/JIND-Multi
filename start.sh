#!/bin/bash

echo "Seleccione la opción que desea ejecutar:"
echo "1) Iniciar la aplicación con Gunicorn"
echo "2) Ejecutar script de bash personalizado"
echo "3) Ambos"

read -p "Ingrese el número de la opción: " opcion

case $opcion in
    1)
        echo "Iniciando la aplicación con Gunicorn..."
        exec gunicorn --bind 0.0.0.0:5003 index:app
        ;;
    2)
        echo "Ejecutando el script de bash..."
        # exec python run-jind-multi --config "/app/config.json"
        # modificar para recibir los parametros de config.json PENDIENTE
        exec python run-jind-multi --config "config.json"
        ;;
        echo "Opción no válida."
        ;;
esac
