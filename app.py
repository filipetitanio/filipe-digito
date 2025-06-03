import streamlit as st
import numpy as np
from PIL import Image
import joblib
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# Configura página
st.set_page_config(page_title="Reconhecimento de Dígitos - Filipe Tchivela", layout="wide")

# Verifica modelo
model_path = 'mnist_model_final_rbf.pkl'
if not os.path.exists(model_path):
    st.error(f"Erro: Arquivo '{model_path}' não encontrado.")
    st.stop()

# Carrega modelo
try:
    model = joblib.load(model_path)
    st.write("Modelo SVM (kernel RBF) carregado com precisão de ~96,83%")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {str(e)}")
    st.stop()

# Carrega subconjunto de teste
data_path = 'mnist_test_subset.csv'
if os.path.exists(data_path):
    test_data = pd.read_csv(data_path)
    X_test = test_data.drop('label', axis=1).to_numpy()
    y_test = test_data['label'].to_numpy()
else:
    st.warning(f"Arquivo '{data_path}' não encontrado. Visualizações limitadas.")
    X_test = None
    y_test = None

# Título
st.title("Reconhecimento de Dígitos Manuscritos")

# Sobre Mim
st.header("Sobre Mim")
st.write("""
- **Nome**: Filipe Tchivela
- **Número de Estudante**: 2022142100
- **Curso**: Ciência da Computação
- **Instituição**: UMN-ISPH
- **Contacto**: +946715031
- **E-mail**: filipetchivela@gmail.com
""")

# Desenhar dígito
st.header("Desenhar um Dígito")
st.write("Desenhe um dígito (0-9) no canvas (28x28 pixels).")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=2,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=28,
    width=28,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    image_array = np.array(image.resize((28, 28))).reshape(1, -1) / 255.0
    prediction = model.predict(image_array)[0]
    st.image(image, caption="Dígito Desenhado", width=100)
    st.write(f"Dígito Previsto: **{prediction}**")

# Carregar imagem
st.header("Carregar uma Imagem")
uploaded_file = st.file_uploader("Carregue uma imagem 28x28 (PNG/JPG)", type=["png", "jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image_array = np.array(image.resize((28, 28))).reshape(1, -1) / 255.0
    prediction = model.predict(image_array)[0]
    st.image(image, caption="Imagem Carregada", width=100)
    st.write(f"Dígito Previsto: **{prediction}**")

# Matriz de confusão
if X_test is not None and y_test is not None:
    st.header("Matriz de Confusão")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Rótulo Previsto')
    ax.set_ylabel('Rótulo Verdadeiro')
    ax.set_title('Matriz de Confusão')
    st.pyplot(fig)

# Exemplos de previsões
if X_test is not None and y_test is not None:
    st.header("Exemplos de Previsões")
    st.write("5 imagens do conjunto de teste:")
    cols = st.columns(5)
    for i in range(5):
        image = X_test[i].reshape(28, 28)  # Já está em [0, 1]
        true_label = y_test[i]
        pred_label = model.predict(X_test[i].reshape(1, -1))[0]
        with cols[i]:
            st.image(image, caption=f"Verd.: {true_label}\nPrev.: {pred_label}", width=80)
