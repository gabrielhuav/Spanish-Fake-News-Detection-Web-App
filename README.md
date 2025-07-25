# Detección de Fraude Digital y Noticias Falsas Usando un Modelo de Lenguaje

Este proyecto presenta una solución usando un Modelo de Lenguaje para la detección de fraude digital y noticias falsas en español. El sistema se compone de dos fases principales:
1.  **Entrenamiento del Modelo:** Se creo un script de Python que unifica múltiples corpus de noticias en español, calibra hiperparámetros y entrena un modelo de lenguaje con una GPU y `DistilBERT` para la clasificación de textos.
2.  **Aplicación Web de Análisis:** Una aplicación web construida con Flask y Docker que permite a un usuario introducir la URL de una noticia para que el modelo entrenado la analice y emita un veredicto en tiempo real.

## Descripción del Proyecto

El objetivo de esta investigación es aplicar técnicas de Procesamiento del Lenguaje Natural (PLN) y aprendizaje profundo para construir una herramienta capaz de discernir entre noticias reales y fraudulentas. El proceso incluye la recopilación y unificación de datos, el ajuste fino (*fine-tuning*) de un modelo Transformer pre-entrenado y la implementación de una aplicación web interactiva para su uso práctico.

## Características

* **Unificación de Corpus:** Consolida múltiples datasets de noticias en español en un único corpus limpio y estandarizado.
* **Modelo Multilingüe:** Utiliza `distilbert-base-multilingual-cased`, un modelo potente y eficiente capaz de entender múltiples idiomas.
* **Calibración de Hiperparámetros:** Emplea `KerasTuner` para encontrar la configuración óptima del modelo, maximizando su rendimiento.
* **Evaluación Robusta:** Divide los datos en conjuntos de entrenamiento (80%), validación (10%) y pruebas (10%) para una evaluación imparcial del modelo.
* **Métricas Completas:** Genera un reporte detallado con Exactitud, Precisión, Exhaustividad (Recall) y F1-Score.
* **Visualización de Resultados:** Crea gráficas de las curvas de aprendizaje y una matriz de confusión para un análisis visual del rendimiento.
* **Aplicación Web Interactiva:** Una interfaz web sencilla donde se puede pegar una URL para obtener un análisis instantáneo.
* **Contenerización con Docker:** Todo el proyecto está dockerizado, garantizando una ejecución fácil y reproducible en cualquier sistema.

## Tecnologías Utilizadas

* **Lenguaje:** Python 3.8
* **Machine Learning:** TensorFlow 2.4, Transformers (Hugging Face), KerasTuner, Scikit-learn
* **Procesamiento de Datos:** Pandas, NumPy
* **Framework Web:** Flask
* **Contenerización:** Docker, Docker Compose
* **Visualización:** Matplotlib, Seaborn

## Estructura del Proyecto
|
|-- app/
|   |-- main.py             # La aplicación Flask
|   |-- templates/
|   |   |-- index.html      # La interfaz de usuario
|   |
|   |-- modelo_final_distilbert_es/ # <-- Tu modelo y tokenizador guardados
|       |-- tf_model.h5
|       |-- config.json
|       |-- tokenizer.json
|       |-- vocab.txt
|       |-- ... (otros archivos)
|
|-- Dockerfile              # La receta para construir el contenedor
|-- docker-compose.yml      # El orquestador para ejecutar la app
|-- requirements.txt        # La lista de dependencias de Python