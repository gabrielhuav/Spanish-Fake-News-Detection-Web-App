# Empezar desde una imagen base de Python 3.9
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Actualizar pip antes de instalar dependencias
RUN pip install --upgrade pip

# Copiar el archivo de requerimientos
COPY requirements.txt requirements.txt

# Instalar todas las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar tu aplicación (código, plantillas y el modelo)
COPY ./app .

# Exponer el puerto
EXPOSE 5000

# Comando para iniciar la aplicación
CMD ["python", "main.py"]