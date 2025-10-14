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

# --- TRADUCCIONES ---
translations = {
    'es': {
        'title': 'Analizador de Fraude Digital',
        'title_english': 'Spanish Fake News Analyzer',
        'url_placeholder': 'Pega una URL de noticia aquí...',
        'text_title_placeholder': 'Pega el titular de la noticia aquí...',
        'text_content_placeholder': 'Pega el contenido de la noticia aquí...',
        'analyze_button': 'Analizar',
        'results_title': 'Resultados del Análisis',
        'verdict_real': 'REAL',
        'verdict_fake': 'FALSO',
        'verdict_insufficient': 'INSUFICIENTE TEXTO',
        'verdict_error': 'ERROR DE ANÁLISIS',
        'confidence_label': 'Confianza',
        'analyzed_text_label': 'Ver el Texto Exacto Analizado por el Modelo',
        'preview_title': 'Vista Previa del Sitio',
        'preview_warning': 'Advertencia: La vista previa de abajo es un iframe del sitio externo. No interactúes con él si no confías en la fuente.',
        'html_content_label': 'Ver Contenido HTML Completo'
    },
    'en': {
        'title': 'Spanish Fake News Analyzer',
        'title_english': 'Spanish Fake News Analyzer',
        'url_placeholder': 'Paste a news URL here...',
        'text_title_placeholder': 'Paste the news headline here...',
        'text_content_placeholder': 'Paste the news content here...',
        'analyze_button': 'Analyze',
        'results_title': 'Analysis Results',
        'verdict_real': 'REAL',
        'verdict_fake': 'FAKE',
        'verdict_insufficient': 'INSUFFICIENT TEXT',
        'verdict_error': 'ANALYSIS ERROR',
        'confidence_label': 'Confidence',
        'analyzed_text_label': 'See the Exact Text Analyzed by the Model',
        'preview_title': 'Site Preview',
        'preview_warning': 'Warning: The preview below is an iframe of the external site. Do not interact with it if you do not trust the source.',
        'html_content_label': 'View Full HTML Content'
    }
}

def extraer_texto_url(url: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        titulo = soup.find('h1').get_text(strip=True) if soup.find('h1') else (soup.find('title').get_text(strip=True) if soup.find('title') else "")
        
        parrafos = soup.find_all('p')
        texto_completo = ' '.join([' '.join(p.stripped_strings) for p in parrafos])
        
        return titulo, texto_completo, str(soup)
    except Exception as e:
        print(f"Error extrayendo texto: {e}")
        return None, None, None

def predecir_noticia(titulo: str, texto: str):
    if not texto or not texto.strip():
        return "INSUFFICIENTE TEXTO", "0.00%", ""

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
    lang = request.args.get('lang', 'es')
    if lang not in translations:
        lang = 'es'

    if request.method == "POST":
        veredicto = None
        confianza = None
        texto_analizado = None
        contenido_html = None
        url_analizada = None

        if 'url' in request.form:
            url = request.form.get("url")
            if url:
                titulo, texto, contenido_html = extraer_texto_url(url)
                
                if titulo is not None:
                    veredicto, confianza, texto_analizado = predecir_noticia(titulo, texto)
                    url_analizada = url
                else:
                    veredicto, confianza, contenido_html = "ERROR DE ANÁLISIS", "N/A", "No se pudo obtener el contenido de la URL."
                    texto_analizado = "N/A"
        
        elif 'text_title' in request.form and 'text_content' in request.form:
            titulo = request.form.get("text_title")
            texto = request.form.get("text_content")
            veredicto, confianza, texto_analizado = predecir_noticia(titulo, texto)

        return render_template("index.html", 
                               veredicto=veredicto, 
                               confianza=confianza, 
                               url_analizada=url_analizada, 
                               texto_completo_html=contenido_html,
                               texto_analizado_modelo=texto_analizado,
                               lang=lang,
                               t=translations[lang])
    
    return render_template("index.html", lang=lang, t=translations[lang])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)