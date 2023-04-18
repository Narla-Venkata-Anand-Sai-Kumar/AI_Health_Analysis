import io
import os
import numpy as np
import streamlit as st
import requests
from PIL import Image
from model import classify
import cv2

@st.cache(allow_output_mutation=True)
# def get_model():
#     return bone_frac()

# pred_model = get_model()
# pred_model=bone_frac()

def predict():
    c=classify('tmp.jpg')
    st.markdown('#### Predicted Captions:')
    st.write(c)

st.title('Image Captioner')
img_url = st.text_input(label='Enter Image URL')

if (img_url != "") and (img_url != None):
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.convert('RGB')
    st.image(img)
    img.save('tmp.jpg')
    predict()
    os.remove('tmp.jpg')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# st.markdown('<center style="opacity: 70%">OR</center>', unsafe_allow_html=True)
img_upload = st.file_uploader(label='Upload Image', type=['jpg', 'png', 'jpeg'])

if img_upload != None:
    img = img_upload.read()
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    img=np.asarray(img)
    print(img)
    # img=cv2.imread(img)
    # img.save('tmp.jpg')
    st.image(img)
    c=classify(img)
    st.markdown('#### Predicted Captions:')
    st.write(c)
    # predict()
    # os.remove('tmp.jpg')