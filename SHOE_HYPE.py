from turtle import onclick
import streamlit as st
import os
import pandas as pd
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import cv2
import webbrowser
from bokeh.models.widgets import Div

hide_menu_style ="""
    <style>
    footer{visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html =True)

if "button_clicked" not in st.session_state:
  st.session_state.button_clicked = False
def callback():
  st.session_state.button_clicked = True

st.title('SHOE HYPE〽️')
st.subheader("FOR THE ONE's ADDICTED TO SNEAKERS👟")

cola,colb=st.columns(2)
with cola:
  print(" ")
with colb:
  st.markdown("**©️COPYRIGHT @DHRUV_TYAGI**")

col13, col14 = st.columns(2)
with col13:
  st.image("https://media3.giphy.com/media/ejcoQeKAnFab6/200w.webp?cid=ecf05e4703ksn0xzhw3200woparv24uubirsfyue6yk3u7m0&rid=200w.webp&ct=g")
  if st.button('ABOUT THE WEBSITE'):
    st.markdown("HOLA EVERYONE✌️ WELCOME TO THE **SHOE HYPE〽️. JUST UPLOAD THE IMAGE OF SHOE AND GET RECOMMENDATIONS BASED ON IT.** THIS IS A **CNN** BASED RECOMMENDER SYSTEM WHICH USES **RESNET** FOR FEATURE EXTRACTION. THE FEATURES OF UPLOADED IMAGE ARE COMPARED WITH THE HELP OF **SCIKIT LEARN**. THEN THE IMAGES ARE RECOMMENDED AND THE ACCOMPANIED DATA IS FETCHED FROM THE DATASET. DEVELOPED BY - **DHRUV TYAGI**.")
  if (
    st.button(' CONNECT WITH US.', on_click=callback)
  or st.session_state.button_clicked):

    st.markdown('**DHRUV TYAGI**')
    st.markdown('📞 : 7983061818' )
    st.markdown('📧 : dhruvtyagionly1@gmail.com')
    if st.button('LINKEDIN🔗'):
      js = "window.open('https://www.linkedin.com/in/dhruv-tyagi-9a526b218')"  
      js = "window.location.href = 'https://www.linkedin.com/in/dhruv-tyagi-9a526b218'" 
      html = '<img src onerror="{}">'.format(js)
      div = Div(text=html)
      st.bokeh_chart(div)
    if st.button('INSTAGRAM🔗'):
      js = "window.open('https://www.instagram.com/iamdhruv.tyagi/')"  
      js = "window.location.href = 'https://www.instagram.com/iamdhruv.tyagi/'" 
      html = '<img src onerror="{}">'.format(js)
      div = Div(text=html)
      st.bokeh_chart(div)

with col14:
      st.image('https://upload.wikimedia.org/wikipedia/commons/f/fa/Bally_Ascar_shoe.gif')


feature_list = np.array(pickle.load(open('embeddings_shoes.pkl','rb')))
filenames = pickle.load(open('filenames_shoes.pkl','rb'))

model = ResNet50 (weights='imagenet', include_top=False, input_shape=(224, 224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file (uploaded_file):
  try:
      with open (os.path.join('C:/PROGRAMMING/PYTHON LANGUAGE/projects/SHOE-HYPE-PROJECT/uploads', uploaded_file.name),'wb') as f:
        f.write(uploaded_file.getbuffer())
      return 1
  except:
      return 0

def extract_features(img_path, model):
    img = image.load_img(img_path,target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    
    return normalized_result

def recommend(features,feature_list):
  neighbors = NearestNeighbors(n_neighbors=5, algorithm ='brute', metric='euclidean')
  neighbors.fit(feature_list)
  distances, indices = neighbors.kneighbors([features])
  return indices

product = pd.read_csv('C:/PROGRAMMING/PYTHON LANGUAGE/projects/SHOE-HYPE-PROJECT/PROJECT SHOES DATASET_CSV.csv')
data = os.listdir("C:/PROGRAMMING/PYTHON LANGUAGE/projects/SHOE-HYPE-PROJECT/SHOES IMAGES")

st.subheader("INSTRUCTIONS:")
st.markdown("FOR BETTER RECOMMENDATIONS UPLOAD THE IMAGES WHICH ONLY HAVE SHOES AND HAVE WHITE BACKGROUND.")
uploaded_file = st.file_uploader("Choose an image") 
if uploaded_file is not None:
  if save_uploaded_file(uploaded_file):
    display_image = Image.open(uploaded_file)

    features = extract_features(os.path.join('C:/PROGRAMMING/PYTHON LANGUAGE/projects/SHOE-HYPE-PROJECT/uploads',uploaded_file.name),model)

    indices=recommend(features,feature_list)
    value = indices.tolist()[0]

    ls=[]
    for i in value:
      ls.append(data[i])
      i=+1
    
    d=[]
    for i in ls:
      try :
        ref=product['REFERENCE'].tolist()
        ind = ref.index(i)
        d.append(product['DESCRIPTION'][ind])
      except :
        d.append('INFORMATION NOT FOUND IN DATABASE')
      i=+1
      

    u=[]
    for i in ls:
      try :
        ref=product['REFERENCE'].tolist()
        ind = ref.index(i)
        u.append(product['PAGE URL'][ind])
      except:
        u.append("INFORMATION NOT FOUND IN DATABASE")
      i=+1

    col11, col12 = st.columns(2)
    with col11:
      basewidth = 180
      wpercent = (basewidth / float(display_image.size[0]))
      hsize = int((float(display_image.size[1]) * float(wpercent)))
      img1 = display_image.resize((basewidth, hsize), Image.ANTIALIAS)
      st.image(img1)

    with col12:
      st.markdown('**UPLOADED IMAGE**')

    if st.button('SHOW RECOMMENDATIONS'):
  
        tab1, tab2, tab3,tab4,tab5 = st.tabs(["TAB 1", "TAB 2", "TAB 3",'TAB 4','TAB 5'])

        with tab1:
          col1, col2= st.columns(2)
          with col1:
            basewidth = 180
            img =Image.open(filenames[indices[0][0]].replace('\\','/'))
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
            st.image(img)
          with col2:
            v=(product['DESCRIPTION'][ref.index(ls[0])]).split(" ")[0:3]
            st.title(' '.join([str(elem) for elem in v]))
            st.markdown(product['PAGE URL'][ref.index(ls[0])], unsafe_allow_html=True)

        with tab2:           
          col3, col4= st.columns(2)
          with col3:
            basewidth = 180
            img =Image.open(filenames[indices[0][1]].replace('\\','/'))
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
            st.image(img)
          with col4:
            v1=(product['DESCRIPTION'][ref.index(ls[1])]).split(" ")[0:3]
            st.title(' '.join([str(elem) for elem in v1]))
            st.markdown(product['PAGE URL'][ref.index(ls[1])], unsafe_allow_html=True)
        
        with tab3:
          col5, col6= st.columns(2)
          with col5:
            basewidth = 180
            img =Image.open(filenames[indices[0][2]].replace('\\','/'))
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
            st.image(img)
          with col6:
            v2=(product['DESCRIPTION'][ref.index(ls[2])]).split(" ")[0:3]
            st.title(' '.join([str(elem) for elem in v2]))
            st.markdown(product['PAGE URL'][ref.index(ls[2])], unsafe_allow_html=True)
          
        with tab4:
          col7, col8= st.columns(2)
          with col7:
            basewidth = 180
            img =Image.open(filenames[indices[0][3]].replace('\\','/'))
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
            st.image(img)
          with col8:
            v3=(product['DESCRIPTION'][ref.index(ls[3])]).split(" ")[0:3]
            st.title(' '.join([str(elem) for elem in v3]))
            st.markdown(product['PAGE URL'][ref.index(ls[3])], unsafe_allow_html=True)
      
        with tab5:
          col9, col10= st.columns(2)
          with col9:
            basewidth = 180
            img =Image.open(filenames[indices[0][4]].replace('\\','/'))
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
            st.image(img)
          with col10:
            v4=(product['DESCRIPTION'][ref.index(ls[4])]).split(" ")[0:3]
            st.title(' '.join([str(elem) for elem in v4]))
            st.markdown(product['PAGE URL'][ref.index(ls[4])], unsafe_allow_html=True)

      
    

      
