import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras import backend as K 
import cv2


st.title('My first app')

st.title("Segmentation  with Dense-UNet")


@st.cache(allow_output_mutation=True)
def loading_model():
    model = load_model('satellitesegment.h5')
    #model._make_predict_function()
    #model.summary()
    session = K.get_session()
    return model,session


@st.cache
def upload_img(image):
    img_npy = np.array(image)
    #img_npy = img_npy.reshape((1,512,512,3))
    
    return img_npy


uploaded_file = st.file_uploader("Choose an image...", type=['tif'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=False)

    button = st.button('Predict')

    if button:
        t = st.empty()
        t.markdown('## İmage is segmenting...')
        model,session = loading_model()
        K.set_session(session)
        image = np.array(image,dtype='uint16')
        t.markdown(f'{type(image)}')
        #image = Image.fromarray(image)
    #     result_img = model.predict(image)
    #     result_img = result_img[:,:,:,:]>0.5
        #result_img = result_img[0,:,:,1]*255
        #t.markdown(f"{result_img}")
        #result_img = Image.fromarray(result_img)
        #t.markdown('## Segmentation result: ')
        #st.image(image, caption='Predicted Image.', use_column_width=False)
