import os
import requests
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# -------------------- Configurações --------------------
DATA_PATH = 'data/phishing_email.csv'
PROCESSED_DATA_PATH = 'data/phishing_email_processed.csv'
TRAIN_DATA_PATH = 'data/train_data.csv'
TEST_DATA_PATH = 'data/test_data.csv'
MODEL_PATH = 'model/phishing_model.h5'
VECTORIZER_PATH = 'model/vectorizer.pkl'
MAX_FEATURES = 10000  # Número máximo de palavras para o vetor TF-IDF
CSV_URL = "https://drive.google.com/uc?export=download&id=1Dxx6m2tuURFkG1AHjaXghjk8sdO3qIc5"  # Altere para o seu link do Google Drive

# -------------------- Funções de Download --------------------
def baixar_csv():
    """
    Verifica se o arquivo CSV está presente localmente e faz o download do Google Drive se necessário.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Arquivo não encontrado. Baixando de {CSV_URL}...")
        response = requests.get(CSV_URL)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            with open(DATA_PATH, 'wb') as f:
                f.write(response.content)
            print(f"Arquivo baixado e salvo em {DATA_PATH}")
        else:
            print("Erro ao baixar o arquivo. Verifique a URL.")
            exit(1)

# -------------------- Funções de Pré-processamento --------------------
def preprocess_data(data_path, save_path):
    """
    Carrega e pré-processa os dados do dataset original.
    Remove colunas desnecessárias, normaliza classes e salva o dataset processado.
    """
    data = pd.read_csv(data_path)
    data = data.rename(columns={'Email Text': 'email_text', 'Email Type': 'email_type'})
    
    # Remove colunas desnecessárias
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    # Remove linhas com valores nulos
    data = data.dropna(subset=['email_text', 'email_type'])
    
    # Normaliza as classes
    data['email_type'] = data['email_type'].map({'Safe Email': 0, 'Phishing Email': 1})
    data = data.dropna(subset=['email_type'])  # Remove linhas inválidas após mapeamento
    
    # Salva o dataset processado
    data.to_csv(save_path, index=False)
    return data

# Função para dividir os dados em treino e teste
def split_data(data, train_path, test_path, test_size=0.2):
    """
    Divide os dados em conjuntos de treino e teste.
    Salva os conjuntos de treino e teste em arquivos separados.
    """
    # Extração dos textos e dos rótulos
    X = data['email_text']
    y = data['email_type']
    
    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Salvar os conjuntos em arquivos CSV
    pd.DataFrame({'email_text': X_train, 'email_type': y_train}).to_csv(train_path, index=False)
    pd.DataFrame({'email_text': X_test, 'email_type': y_test}).to_csv(test_path, index=False)
    
    return X_train, X_test, y_train, y_test

# Função para vetorização dos dados
def vectorize_data(X_train, X_test, max_features):
    """
    Converte os textos em vetores TF-IDF.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# -------------------- Funções de Criação e Treinamento do Modelo --------------------
def create_model(input_dim):
    """
    Define a arquitetura do modelo de rede neural.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=400):
    """
    Treina o modelo com os dados TF-IDF.
    """
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1)
    return model, history

# -------------------- Salvar Modelo e Vectorizer --------------------
def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    """
    Salva o modelo treinado e o vectorizer TF-IDF.
    """
    model.save(model_path)
    with open(vectorizer_path, 'wb') as file:
        pickle.dump(vectorizer, file)

# -------------------- Fluxo Principal --------------------
if __name__ == '__main__':
    # 1. Baixar o arquivo CSV se necessário
    baixar_csv()

    # 2. Pré-processar os dados
    data = preprocess_data(DATA_PATH, PROCESSED_DATA_PATH)
    
    # 3. Dividir os dados
    X_train, X_test, y_train, y_test = split_data(data, TRAIN_DATA_PATH, TEST_DATA_PATH)
    
    # 4. Vetorizar os dados
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_data(X_train, X_test, MAX_FEATURES)
    
    # 5. Criar o modelo
    input_dim = X_train_tfidf.shape[1]
    model = create_model(input_dim)
    
    # 6. Treinar o modelo
    model, history = train_model(model, X_train_tfidf, y_train, X_test_tfidf, y_test, epochs=2)
    
    # 7. Salvar o modelo e o vectorizer
    save_model_and_vectorizer(model, vectorizer, MODEL_PATH, VECTORIZER_PATH)
    
    print("Treinamento concluído. Modelo e vectorizer salvos com sucesso!")
