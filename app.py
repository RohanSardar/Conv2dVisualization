import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title='Convolution Visualizer', page_icon='🏞️', initial_sidebar_state='expanded', layout='wide')
st.title('Convolution Feature Map Visualizer')
st.badge('It uses Conv2d() method of PyTorch', icon='ℹ️', color='orange')

uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

st.sidebar.header('Conv2d Parameters')

out_channels = st.sidebar.slider('Out Channels', 4, 64, 16)
kernel_size = st.sidebar.slider('Kernel Size', 1, 16, 1)
stride = st.sidebar.slider('Stride', 1, 8, 1)
padding = st.sidebar.slider('Padding', 0, 8, 1)

process = st.sidebar.button('Process Image')

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=False)
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0)

    conv = nn.Conv2d(
        in_channels=3,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

elif not uploaded_file:
    st.warning('Upload an image to begin.')

if uploaded_file and process:
    with torch.no_grad():
        out = conv(img_tensor)

    num_features = out.shape[1]

    cols = 4
    rows = math.ceil(num_features / cols)

    with st.spinner('Processing convolution...', show_time=True):
        fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
        axes = axes.flatten()

        for i in range(num_features):
            axes[i].imshow(out[0, i].numpy(), cmap='gray')
            axes[i].set_title(f'Feature {i+1}')
            axes[i].axis('off')

        for j in range(num_features, len(axes)):
            axes[j].axis('off')

        st.pyplot(fig)

footer_html = """<div style='text-align: center;'>
  <p style="font-size:80%; font-family: 'Trebuchet MS';">
  Developed by <a href="https://linktr.ee/RohanSardar">Rohan Sardar</a>
  <br>Project completed on 2nd December 2025</p>
</div>"""
st.sidebar.markdown(footer_html, unsafe_allow_html=True)
