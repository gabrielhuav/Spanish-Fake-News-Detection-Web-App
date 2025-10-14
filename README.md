# Fake News Detection Using a Language Model

This project presents a solution using a Language Model for the detection of fake news in Spanish. The system is composed of two main phases:

1.  **Model Training:** A Python script was created that unifies multiple news corpora in Spanish, calibrates hyperparameters, and trains a language model with a GPU and `DistilBERT` for text classification.
2.  **Web Analysis Application:** A web application built with Flask and Docker that allows a user to enter the URL of a news article for the trained model to analyze and issue a real-time verdict.

## Project Description

The objective of this research is to apply Natural Language Processing (NLP) and deep learning techniques to build a tool capable of discerning between real and fake news. The process includes data collection and unification, fine-tuning of a pre-trained Transformer model, and the implementation of an interactive web application for practical use.

## System Architecture

### Components Diagram
<img width="1716" height="1076" alt="image" src="https://github.com/user-attachments/assets/1541ba6f-0ea5-4eee-9823-f9f405c17255" />

The system consists of three main components:
- **Web Interface (Flask):** Handles user requests and displays results
- **Text Processor:** Extracts and preprocesses text from news URLs
- **Classification Model (DistilBERT):** Analyzes the text and predicts authenticity

### Sequence Diagram

<img width="1477" height="1290" alt="image" src="https://github.com/user-attachments/assets/9e0946fe-37a8-408b-acf9-b2fdc2f39a97" />

The diagram above illustrates the complete flow from when a user submits a URL until they receive the classification result, showing the interaction between all system components.

## Related Repository

For the complete model training process, dataset preparation, and hyperparameter tuning, please visit the training repository:

**🔗 [Spanish Fake News Detection - Training](https://github.com/gabrielhuav/Spanish-Fake-News-Detection-Training)**

This repository contains:
- Data collection and preprocessing scripts
- Model training pipeline with KerasTuner
- Evaluation metrics and visualizations
- Dataset unification from multiple Spanish news corpora

## Features

* **Corpus Unification:** Consolidates multiple news datasets in Spanish into a single clean and standardized corpus.
* **Multilingual Model:** Uses `distilbert-base-multilingual-cased`, a powerful and efficient model capable of understanding multiple languages.
* **Hyperparameter Calibration:** Employs `KerasTuner` to find the optimal model configuration, maximizing its performance.
* **Robust Evaluation:** Splits the data into training (80%), validation (10%), and testing (10%) sets for an unbiased evaluation of the model.
* **Complete Metrics:** Generates a detailed report with Accuracy, Precision, Recall, and F1-Score.
* **Results Visualization:** Creates learning curve graphs and a confusion matrix for a visual analysis of performance.
* **Interactive Web Application:** A simple web interface where you can paste a URL to get an instant analysis.
* **Containerization with Docker:** The entire project is dockerized, ensuring easy and reproducible execution on any system.

## Technologies Used

* **Language:** Python 3.8
* **Machine Learning:** TensorFlow 2.4, Transformers (Hugging Face), KerasTuner, Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Web Framework:** Flask
* **Containerization:** Docker, Docker Compose
* **Visualization:** Matplotlib, Seaborn

## Project Structure
```
.
├── app/
│   ├── main.py                      # The Flask application
│   ├── templates/
│   │   └── index.html               # The user interface
│   │
│   └── modelo_final_distilbert_es/  # Your saved model and tokenizer
│       ├── tf_model.h5
│       ├── config.json
│       ├── tokenizer.json
│       ├── vocab.txt
│       └── ...                      # Other model files
│
├── images/                          # Documentation images
│   ├── components-diagram.png
│   └── sequence-diagram.png
│
├── Dockerfile                       # The recipe to build the container
├── docker-compose.yml               # The orchestrator to run the app
└── requirements.txt                 # The list of Python dependencies
```
