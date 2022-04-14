import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
from tensorflow.keras.models import load_model
st.write("""
          # Malaria Detection
          """
          )
upload_file = st.sidebar.file_uploader("Upload Cell Images", type="png")
Generate_pred=st.sidebar.button("Predict")
model=load_model('D:/Anaconda3/envs/densenet_model')

def load_and_preprocess_test_images(img):

  #img = cv2.resize(img, img_size, cv2.INTER_CUBIC)
  img_copy = img.copy()
  img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
  img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
  edges = cv2.Canny(img_copy, threshold1 = 80, threshold2 = 160)
  edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
  final_img = cv2.addWeighted(img, 0.5, edges, 0.5, 0)
  final_img = np.expand_dims(final_img, axis = 0)
  final_img = final_img/255.

  return final_img
  
   
def import_n_pred(image_data, model):
    
    size = (64,64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = load_and_preprocess_test_images(img)
    pred = model.predict(img)

    return pred
    
if Generate_pred:
    image=Image.open(upload_file)
    with st.expander('Cell Image', expanded = True):
        st.image(image, use_column_width=True)
    pred=import_n_pred(image, model)
    class_indices = np.argmax(pred, axis = 1)
    if class_indices == 0:
        class_label = 'Uninfected'

    else:
        class_label = 'Parasitized'
 
    st.title("Prediction of image is {}".format(class_label))
    
    



