import streamlit as st
from zipfile import ZipFile
from PIL import Image
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras import backend as K 
import cv2
import tempfile


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
    
    #image = Image.open(uploaded_file)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    image = cv2.imread(tfile.name,-1)
    t_img = Image.fromarray(image/2**3)
    st.image(t_img, caption='Uploaded Image.', use_column_width=False)

    button = st.button("Let's Predict Image")

    if button:
        t = st.empty()
        #t.markdown('## İmage is segmenting...')
        t.markdown(f'{image.shape}')
        """ model,session = loading_model()
        K.set_session(session)
        image = np.array(image,dtype='uint16').reshape((1,512,512,3))
        result_img = model.predict(image)
        result_img = result_img[:,:,:,:]>0.5
        result_img = result_img[0,:,:,0]
        result_img = Image.fromarray(result_img)
        t.markdown('## Segmentation result: ')
        st.image(result_img, caption='Predicted Image.', use_column_width=False) """
