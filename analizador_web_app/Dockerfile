# Empezar desde una imagen base de Python compatible con las librerías
FROM python:3.8-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requerimientos primero
COPY requirements.txt requirements.txt

# Instalar todas las dependencias fijadas del archivo
RUN pip install --no-cache-dir -r requirements.txt

# Copiar tu aplicación (código, plantillas y el modelo)
COPY ./app .

# Exponer el puerto en el que correrá la aplicación
EXPOSE 5000

# Comando para iniciar la aplicación Flask
CMD ["python", "main.py"]