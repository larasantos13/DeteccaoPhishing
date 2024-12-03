from flask import Flask, render_template, request, jsonify
import pickle
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re

# -------------------- Configurações do Flask --------------------
app = Flask(__name__)

# -------------------- Funções de Pré-processamento --------------------
def normalize_large_text(text, max_length=20000):
    """
    Normaliza um texto grande, removendo espaços excessivos
    e limitando o número de caracteres.
    """
    text = ' '.join(text.split())  # Remove espaços duplicados
    return text[:max_length]  # Limita ao máximo de caracteres permitidos


def split_text_into_chunks(text, chunk_size=500):
    """
    Divide um texto em pedaços menores com tamanho máximo definido.
    """
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# -------------------- Carregamento do Modelo e Vectorizer --------------------
def load_model_and_vectorizer():
    """
    Carrega o modelo e o vetor TF-IDF previamente treinados.
    """
    model = tf.keras.models.load_model('model/phishing_model.h5')
    with open('model/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


# -------------------- Configurações Iniciais --------------------
model, vectorizer = load_model_and_vectorizer()


# -------------------- Rotas do Flask --------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Rota principal para análise de textos de emails.
    """
    if request.method == 'POST':
        email_text = request.form['email_text']
        
        # Verificação básica do texto
        if not email_text.strip():
            return render_template('index.html', result="Texto inválido!", email_text=email_text)
        
        # Normalização e processamento
        email_text = normalize_large_text(email_text)
        
        try:
            chunks = split_text_into_chunks(email_text, chunk_size=500)
            chunk_predictions = [
                model.predict(vectorizer.transform([chunk]))[0][0] for chunk in chunks
            ]
            # Decisão final: qualquer bloco >= 0.5 é phishing
            final_result = 'Phishing' if any(pred >= 0.5 for pred in chunk_predictions) else 'Safe'
        except Exception as e:
            final_result = f"Erro ao processar: {str(e)}"
        
        return render_template('index.html', result=final_result, email_text=email_text)
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API para análise de texto de emails via JSON.
    """
    try:
        email_text = request.json['email']
        email_text = normalize_large_text(email_text)
        chunks = split_text_into_chunks(email_text, chunk_size=500)
        chunk_predictions = [
            model.predict(vectorizer.transform([chunk]))[0][0] for chunk in chunks
        ]
        final_result = "Phishing" if any(pred >= 0.5 for pred in chunk_predictions) else "Safe"
        return jsonify({"message": final_result})
    except Exception as e:
        return jsonify({"error": f"Erro ao processar: {str(e)}"}), 500


# -------------------- Inicialização do Servidor --------------------
if __name__ == '__main__':
    app.run(debug=True)

