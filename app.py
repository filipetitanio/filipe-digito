# Importando bibliotecas necessárias
import streamlit as st
import numpy as np
from PIL import Image
import joblib
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Configurando a página
st.set_page_config(page_title="Reconhecimento de Dígitos Manuscritos", layout="wide")

# Carregando o modelo treinado
try:
    model = joblib.load('mnist_model_final_rbf.pkl')
    st.write("Modelo SVM (kernel RBF) carregado com precisão de 96,99%")
except FileNotFoundError:
    st.error("Erro: Arquivo 'mnist_model_final_rbf.pkl' não encontrado.")
    st.stop()

# Carregando um subconjunto de dados de teste
try:
    test_data = pd.read_csv('mnist_test_subset.csv')  # Subconjunto com 100 amostras
    X_test = test_data.drop('label', axis=1).values / 255.0  # Normaliza
    y_test = test_data['label'].values
except FileNotFoundError:
    st.warning("Arquivo 'mnist_test_subset.csv' não encontrado. Algumas visualizações estarão indisponíveis.")
    X_test = None
    y_test = None

# Título da aplicação
st.title("Reconhecimento de Dígitos Manuscritos")

# Seção "Sobre Mim"
st.header("Sobre Mim")
st.write("""
- **Nome**: Filipe Tchivela
- **Número de Estudante**: 2022142100
- **Curso**: Ciência da Computação
- **Instituição**: UMN-ISPH
- **Contacto**: +946715031
- **E-mail**: filipetchivela@gmail.com
""")

# Seção para desenhar um dígito
st.header("Desenhar um Dígito")
st.write("Desenhe um dígito no canvas abaixo (28x28 pixels). Use o pincel branco sobre fundo preto.")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # Pincel branco
    stroke_width=2,
    stroke_color="#FFFFFF",
    background_color="#000000",  # Fundo preto
    height=28,
    width=28,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Pré-processando a imagem do canvas
    image = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).reshape(1, -1) / 255.0

    # Fazendo a previsão
    prediction = model.predict(image_array)[0]
    
    # Exibindo a imagem e a previsão
    st.image(image, caption="Dígito Desenhado", width=100)
    st.write(f"Dígito Previsto: **{prediction}**")

# Seção para carregar uma imagem
st.header("Carregar uma Imagem")
uploaded_file = st.file_uploader("Carregue uma imagem 28x28 em escala de cinza (PNG ou JPG)", type=["png", "jpg"])

if uploaded_file is not None:
    # Carregando e pré-processando a imagem
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).reshape(1, -1) / 255.0

    # Fazendo a previsão
    prediction = model.predict(image_array)[0]
    
    # Exibindo a imagem e a previsão
    st.image(image, caption="Imagem Carregada", width=100)
    st.write(f"Dígito Previsto: **{prediction}**")

# Seção para matriz de confusão
if X_test is not None and y_test is not None:
    st.header("Matriz de Confusão")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Gerando a matriz de confusão com seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Rótulo Previsto')
    ax.set_ylabel('Rótulo Verdadeiro')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)

# Seção para exemplos de previsões
if X_test is not None and y_test is not None:
    st.header("Exemplos de Previsões")
    st.write("5 imagens do conjunto de teste com rótulos verdadeiros e previstos:")
    cols = st.columns(5)
    for i in range(5):
        image = X_test[i].reshape(28, 28) * 255  # Desnormaliza
        true_label = y_test[i]
        pred_label = model.predict(X_test[i].reshape(1, -1))[0]
        with cols[i]:
            st.image(image, caption=f"Verd.: {true_label}\nPrev.: {pred_label}", width=80)
