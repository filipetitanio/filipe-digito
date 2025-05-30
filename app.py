# Importando bibliotecas necessárias para a aplicação Streamlit
import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Configurando a página da aplicação
st.set_page_config(page_title="Reconhecimento de Dígitos Manuscritos", layout="wide")

# Carregando o modelo treinado
model = joblib.load('mnist_model_final_rbf.pkl')
st.write("Modelo SVM (kernel RBF) carregado com precisão de 96,99%")

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

# Seção para carregar e prever dígitos
st.header("Prever um Dígito")
uploaded_file = st.file_uploader("Carregue uma imagem 28x28 em escala de cinza (PNG ou JPG)", type=["png", "jpg"])

if uploaded_file is not None:
    # Carregando e pré-processando a imagem
    image = Image.open(uploaded_file).convert('L')  # Convertendo para escala de cinza
    image = image.resize((28, 28))  # Redimensionando para 28x28
    image_array = np.array(image).reshape(1, -1) / 255.0  # Normalizando para 0-1

    # Fazendo a previsão
    prediction = model.predict(image_array)[0]
    
    # Exibindo a imagem e a previsão
    st.image(image, caption="Imagem Carregada", width=100)
    st.write(f"Dígito Previsto: **{prediction}**")

# Exibindo uma imagem de exemplo do conjunto de teste
st.header("Exemplo de Previsão")
# Usando a primeira imagem do conjunto de teste (definido na Etapa 2)
sample_image = X_test[0].reshape(28, 28) * 255  # Desnormalizando para exibição
sample_pred = model.predict(X_test[0].reshape(1, -1))[0]
st.image(sample_image, caption=f"Imagem de Exemplo (Rótulo Verdadeiro: {y_test[0]})", width=100)
st.write(f"Dígito Previsto: **{sample_pred}**")
