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
MODELO_PATH = './modelo_pso/'

# --- Cargar todos los componentes del modelo al iniciar ---
print("Cargando modelo de Enjambre de Partículas (PSO)...")
try:
    vectorizer = load(os.path.join(MODELO_PATH, 'vectorizer.joblib'))
    selector = load(os.path.join(MODELO_PATH, 'selector_caracteristicas_pso.joblib'))
    pesos = np.load(os.path.join(MODELO_PATH, 'modelo_pso_pesos.npy'))
    umbrales = np.load(os.path.join(MODELO_PATH, 'modelo_pso_umbrales.npy'))
    
    # En el modelo PSO, la 'solucion' de características es fija, la recreamos
    num_caracteristicas = selector.get_support().sum()
    solucion_fija = np.arange(num_caracteristicas)
    
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('spanish'))
    print("✅ Sistema listo para recibir peticiones.")
except FileNotFoundError:
    print("❌ Error fatal: Faltan archivos del modelo en la carpeta 'app/modelo_pso/'.")
    exit()

# --- Funciones de Procesamiento ---
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

    # 1. Convertir texto a vector BoW
    vector_bow = vectorizer.transform([texto_limpio])
    # 2. Reducir características con el selector
    vector_reducido = selector.transform(vector_bow).toarray()
    
    # 3. Aplicar la lógica de clasificación del modelo PSO
    caracteristicas_activas = vector_reducido[0, solucion_fija]
    x_binario = (caracteristicas_activas >= umbrales).astype(int)
    logit = np.dot(pesos, x_binario)
    
    probabilidad = 1 / (1 + np.exp(-logit))
    
    # La confianza es la probabilidad de ser "REAL"
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

# --- Rutas de la Aplicación ---
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