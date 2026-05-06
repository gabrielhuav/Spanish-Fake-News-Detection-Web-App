# 🕵️‍♂️ Spanish Fake News & Satire Detection System (BETO 3-Class)

🌍 **[🇬🇧 English](#-english) | [🇪🇸 Español](#-español)**

---

## 🇬🇧 English

This repository contains the official implementation of the misinformation detection system described in the paper: *"Systematic Fine-Tuning of Transformer Models for Domain-Specific Misinformation Detection in Spanish Social Media Text"*.

The system leverages a heavily regularized **BETO (Spanish BERT)** model fine-tuned on a 61,674-article unified corpus to classify Spanish news into three distinct categories: **Real, Fake, and Satire**.

### 🔬 System Architecture

The application is deployed using a containerized microservices architecture (Docker), effectively decoupling the web scraping pipeline, the deep learning inference engine, and the user interface.

| Component Diagram | Sequence Diagram |
|---|---|
| <img loading="lazy" src="https://github.com/user-attachments/assets/5290769d-5839-4920-b6c1-1cc1d99585af" alt="System Component Diagram" width="450"/> | <img loading="lazy" src="https://github.com/user-attachments/assets/3cc2f77d-a103-47cf-b1e2-5966d635d195" alt="Inference Sequence Diagram" width="450"/> |

### 🚀 Installation & Deployment

The deployment pipeline is designed to be model-agnostic and environment-independent.

**Prerequisites:**
* Docker & Docker Compose

**1. Clone the repository**
```bash
git clone [https://github.com/gabrielhuav/spanish-fake-news-detection-web-app.git](https://github.com/gabrielhuav/spanish-fake-news-detection-web-app.git)
cd spanish-fake-news-detection-web-app/inference_service
```

**2. Download Model Weights**
Due to file size constraints, the optimized TensorFlow weights (`tf_model.h5` ~500MB) are hosted externally. Download the model artifacts and place them in `app/models/beto_v11_3_classes/`.

> 🔗 **[Download Pre-trained Model Weights (.h5) via Google Drive](https://drive.google.com/file/d/1teOuTTb_4QuFbroBzzmdUH_PHFcy_EOF/view?usp=sharing)**

**3. Run the Container**
```bash
docker-compose up --build -d
```
The web interface will be accessible at `http://localhost:5000`.

### 📊 Detection Use Cases

By isolating satirical content from malicious fake news, the model achieves a 1.0 F1-score on satire, preventing stylistic confounding.

| Real News Case | Fake News Case |
|---|---|
| <img loading="lazy" src="https://github.com/user-attachments/assets/32d53636-6023-4164-a1c1-9d438daf3a20" alt="Real News Detection" width="400"/><br><br>🔗 **Source:** [View Article](https://heraldodemexico.com.mx/nacional/2026/5/6/bts-estara-en-palacio-nacional-asi-lo-anuncio-sheinbaum-en-la-mananera-807807.html) | <img loading="lazy" src="https://github.com/user-attachments/assets/355daeca-f4d2-407f-a247-a0e113e3e3ab" alt="Fake News Detection" width="400"/><br><br>🔗 **Source:** [View Article](https://www.servimedia.es/noticias/bng-ira-toma-posesion-sheinbaum-defiende-no-invite-rey/1410263836) |

| Satirical Content Case |
|---|
| <img loading="lazy" src="https://github.com/user-attachments/assets/7406e584-89b8-4e12-af1b-a25179863987" alt="Satire Detection" width="800"/><br><br>🔗 **Source:** [View Article](https://eldeforma.com/2026/01/20/ice-vehiculos-vandalizados-hondurenos-mexicanos/) |

---
<br>

## 🇪🇸 Español

Este repositorio contiene la implementación oficial del sistema de detección de desinformación descrito en el artículo: *"Systematic Fine-Tuning of Transformer Models for Domain-Specific Misinformation Detection in Spanish Social Media Text"*.

El sistema utiliza un modelo **BETO (Spanish BERT)** con regularización agresiva, ajustado sobre un corpus unificado de 61,674 artículos para clasificar noticias en español en tres categorías: **Real, Falsa y Satírica**.

### 🔬 Arquitectura del Sistema

La aplicación se despliega mediante una arquitectura de microservicios en contenedores (Docker), desacoplando el pipeline de web scraping, el motor de inferencia de Deep Learning y la interfaz de usuario.

| Diagrama de Componentes | Diagrama de Secuencias |
|---|---|
| <img loading="lazy" src="https://github.com/user-attachments/assets/5290769d-5839-4920-b6c1-1cc1d99585af" alt="System Component Diagram" width="450"/> | <img loading="lazy" src="https://github.com/user-attachments/assets/3cc2f77d-a103-47cf-b1e2-5966d635d195" alt="Inference Sequence Diagram" width="450"/> |

### 🚀 Instalación y Despliegue

El pipeline de despliegue está diseñado para ser independiente del entorno y agnóstico al modelo.

**Requisitos previos:**
* Docker y Docker Compose

**1. Clonar el repositorio**
```bash
git clone [https://github.com/gabrielhuav/spanish-fake-news-detection-web-app.git](https://github.com/gabrielhuav/spanish-fake-news-detection-web-app.git)
cd spanish-fake-news-detection-web-app/inference_service
```

**2. Descargar los Pesos del Modelo**
Debido a las restricciones de tamaño, los pesos de TensorFlow optimizados (`tf_model.h5` ~500MB) están alojados externamente. Descarga los artefactos del modelo y colócalos en `app/models/beto_v11_3_classes/`.

> 🔗 **[Descargar Pesos del Modelo (.h5) vía Google Drive](https://drive.google.com/file/d/1teOuTTb_4QuFbroBzzmdUH_PHFcy_EOF/view?usp=sharing)**

**3. Ejecutar el Contenedor**
```bash
docker-compose up --build -d
```
La interfaz web estará disponible en `http://localhost:5000`.

### 📊 Casos de Uso y Detección

Al aislar el contenido satírico de las noticias falsas maliciosas, el modelo alcanza un F1-score de 1.0 en sátira, evitando confusiones estilísticas.


| Caso: Noticia Real | Caso: Noticia Falsa |
|---|---|
| <img loading="lazy" src="https://github.com/user-attachments/assets/32d53636-6023-4164-a1c1-9d438daf3a20" alt="Real News Detection" width="400"/><br><br>🔗 **Fuente:** [Consultar Artículo](https://heraldodemexico.com.mx/nacional/2026/5/6/bts-estara-en-palacio-nacional-asi-lo-anuncio-sheinbaum-en-la-mananera-807807.html) | <img loading="lazy" src="https://github.com/user-attachments/assets/355daeca-f4d2-407f-a247-a0e113e3e3ab" alt="Fake News Detection" width="400"/><br><br>🔗 **Fuente:** [Consultar Artículo](https://www.servimedia.es/noticias/bng-ira-toma-posesion-sheinbaum-defiende-no-invite-rey/1410263836) |

| Caso: Contenido Satírico |
|---|
| <img loading="lazy" src="https://github.com/user-attachments/assets/7406e584-89b8-4e12-af1b-a25179863987" alt="Satire Detection" width="800"/><br><br>🔗 **Fuente** [Consultar Artículo](https://eldeforma.com/2026/01/20/ice-vehiculos-vandalizados-hondurenos-mexicanos/) |
