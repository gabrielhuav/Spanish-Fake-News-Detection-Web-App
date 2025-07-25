import os
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import numpy as np
from flask import Flask, request, render_template

# --- CONFIGURACIÓN ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODELO_GUARDADO_PATH = './modelo_final_distilbert_es'

# --- INICIALIZACIÓN DE LA APP ---
app = Flask(__name__)

# --- CARGA DEL MODELO Y TOKENIZADOR ---
print("Cargando modelo y tokenizador desde el disco...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_GUARDADO_PATH)
model = TFAutoModelForSequenceClassification.from_pretrained(MODELO_GUARDADO_PATH)
print("✅ Sistema listo para recibir peticiones.")

def extraer_texto_url(url: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        titulo = soup.find('h1').get_text(strip=True) if soup.find('h1') else (soup.find('title').get_text(strip=True) if soup.find('title') else "")
        
        # --- CORRECCIÓN CLAVE ---
        # En lugar de .get_text(), usamos el generador .strings y lo unimos con un espacio.
        # Esto preserva los espacios entre diferentes etiquetas HTML (como <b>, <i>, <span>, etc.)
        parrafos = soup.find_all('p')
        texto_completo = ' '.join([' '.join(p.stripped_strings) for p in parrafos])
        # --- FIN DE LA CORRECCIÓN ---
        
        return titulo, texto_completo, str(soup)
    except Exception as e:
        print(f"Error extrayendo texto: {e}")
        return None, None, None

def predecir_noticia(titulo: str, texto: str):
    if not texto or not texto.strip():
        return "INSUFICIENTE TEXTO", "0.00%"

    texto_combinado = titulo + " [SEP] " + texto
    inputs = tokenizer(texto_combinado, return_tensors="tf", truncation=True, padding=True, max_length=256)
    logits = model(inputs).logits
    probabilidades = tf.nn.softmax(logits, axis=1)[0].numpy()
    
    clase_predicha = np.argmax(probabilidades)
    confianza = probabilidades[clase_predicha]
    resultado = "REAL" if clase_predicha == 1 else "FALSO"
    
    return resultado, f"{confianza:.2%}", texto_combinado

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            titulo, texto, contenido_html = extraer_texto_url(url)
            
            if titulo is not None:
                veredicto, confianza, texto_analizado = predecir_noticia(titulo, texto)
            else:
                veredicto, confianza, contenido_html = "ERROR DE ANÁLISIS", "N/A", "No se pudo obtener el contenido de la URL."
                texto_analizado = "N/A"
            
            return render_template("index.html", 
                                   veredicto=veredicto, 
                                   confianza=confianza, 
                                   url_analizada=url, 
                                   texto_completo_html=contenido_html,
                                   texto_analizado_modelo=texto_analizado)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)