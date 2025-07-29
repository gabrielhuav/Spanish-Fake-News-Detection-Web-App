import os
import numpy as np
from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from joblib import load
import re
import nltk
from nltk.corpus import stopwords

# --- CONFIGURACIÓN E INICIALIZACIÓN ---
app = Flask(__name__)
# --- CAMBIO 1: Apuntar a la carpeta del modelo VNS ---
MODELO_PATH = './modelo_vns/'

# --- Cargar todos los componentes del modelo VNS al iniciar ---
print("Cargando modelo de Búsqueda en Vecindades Variables (VNS)...")
try:
    # --- CAMBIO 2: Cargar los archivos específicos de VNS ---
    vectorizer = load(os.path.join(MODELO_PATH, 'vectorizer.joblib'))
    selector = load(os.path.join(MODELO_PATH, 'selector_caracteristicas_vns.joblib'))
    solucion = np.load(os.path.join(MODELO_PATH, 'modelo_vns_solucion.npy'))
    pesos = np.load(os.path.join(MODELO_PATH, 'modelo_vns_pesos.npy'))
    umbrales = np.load(os.path.join(MODELO_PATH, 'modelo_vns_umbrales.npy'))
    
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('spanish'))
    print("✅ Sistema listo para recibir peticiones.")
except FileNotFoundError:
    print("❌ Error fatal: Faltan archivos del modelo en la carpeta 'app/modelo_vns/'.")
    exit()

# --- Funciones de Procesamiento (sin cambios) ---
def limpiar_texto(texto: str):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    tokens = nltk.word_tokenize(texto)
    return " ".join([token for token in tokens if token not in stop_words and len(token) > 2])

def predecir_texto(texto_nuevo: str):
    texto_limpio = limpiar_texto(texto_nuevo)
    if not texto_limpio:
        return "INSUFICIENTE TEXTO", "N/A", "No se encontró texto analizable."

    vector_bow = vectorizer.transform([texto_limpio])
    vector_reducido = selector.transform(vector_bow).toarray()
    
    caracteristicas_activas = vector_reducido[0, solucion]
    x_binario = (caracteristicas_activas >= umbrales).astype(int)
    logit = np.dot(pesos, x_binario)
    
    probabilidad = 1 / (1 + np.exp(-logit))
    
    confianza_str = f"{probabilidad:.2%}" 
    resultado = "REAL" if probabilidad >= 0.5 else "FALSO"
    
    return resultado, confianza_str, texto_limpio

def extraer_texto_url(url: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        titulo = soup.find('h1').get_text(strip=True) if soup.find('h1') else (soup.find('title').get_text(strip=True) if soup.find('title') else "")
        texto_completo = ' '.join([' '.join(p.stripped_strings) for p in soup.find_all('p')])
        
        return titulo, texto_completo, str(soup)
    except Exception as e:
        print(f"Error extrayendo texto: {e}")
        return None, None, None

# --- Rutas de la Aplicación (sin cambios) ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            titulo, texto_parrafos, contenido_html = extraer_texto_url(url)
            
            if texto_parrafos is not None:
                texto_completo = titulo + " " + texto_parrafos
                veredicto, confianza, texto_analizado = predecir_texto(texto_completo)
            else:
                veredicto, confianza, texto_analizado, contenido_html = "ERROR DE ANÁLISIS", "N/A", "N/A", "No se pudo obtener el contenido."
            
            return render_template("index.html", 
                                   veredicto=veredicto, 
                                   confianza=confianza, 
                                   url_analizada=url,
                                   texto_analizado_modelo=texto_analizado,
                                   texto_completo_html=contenido_html)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)