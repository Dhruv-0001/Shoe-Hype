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
  st.image("https://i.pinimg.com/originals/1f/f5/94/1ff594ed96063b9db4866efaaa864ef6.gif")
  if st.button('ABOUT THE WEBSITE'):
    st.markdown("HOLA EVERYONE✌️ WELCOME TO THE **SHOE HYPE〽️. JUST UPLOAD THE IMAGE OF SHOE AND GET RECOMMENDATIONS BASED ON IT.** THIS IS A **CNN** BASED RECOMMENDER SYSTEM WHICH USES **RESNET** FOR FEATURE EXTRACTION. THE FEATURES OF UPLOADED IMAGE ARE COMPARED WITH THE HELP OF **SCIKIT LEARN**. THEN THE IMAGES ARE RECOMMENDED AND THE ACCOMPANIED DATA IS FETCHED FROM THE DATASET. DEVELOPED BY -- @**DHRUV TYAGI**.")
  if (
    st.button('CONNECT WITH US !', on_click=callback)
  or st.session_state.button_clicked):

    st.markdown('**DHRUV TYAGI**')
    st.markdown('📞 +917983061818' )
    st.markdown('🖄 dhruvtyagionly1@gmail.com')
    if st.button('LINKEDIN'):
      webbrowser.open('https://www.linkedin.com/in/dhruv-tyagi-9a526b218/')


with col14:
      st.image("https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/d46df6106232211.5f8b32696e3a7.gif")  
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
      with open (os.path.join('uploads', uploaded_file.name),'wb') as f:
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

product = pd.read_csv('dataset_csv.csv')
data = os.listdir("SHOES_IMAGES")

st.subheader("INSTRUCTIONS:")
st.markdown("__FOR BETTER RECOMMENDATIONS UPLOAD THE IMAGES WHICH ONLY HAVE SHOES AND HAVE WHITE BACKGROUND.__")
uploaded_file = st.file_uploader("Choose an image") 
if uploaded_file is not None:
  if save_uploaded_file(uploaded_file):
    display_image = Image.open(uploaded_file)

    features = extract_features(os.path.join('uploads',uploaded_file.name),model)

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
  
        tab1, tab2, tab3, tab4, tab5 = st.tabs(['TAB1','TAB2','TAB3','TAB4','TAB5'])

        with tab1:
          col1, col2= st.columns(2)
          with col1:
            ok=os.path.join('https://m.media-amazon.com/images/I/', list(filenames[indices[0][0]].split("\\"))[1] )
            st.image(ok)
          with col2:
            v=(product['DESCRIPTION'][ref.index(ls[0])]).split(" ")[0:3]
            st.title((' '.join([str(elem) for elem in v])).upper())
            st.markdown(product['PAGE URL'][ref.index(ls[0])], unsafe_allow_html=True)

        with tab2:           
          col3, col4= st.columns(2)
          with col3:
             ok=os.path.join('https://m.media-amazon.com/images/I/', list(filenames[indices[0][1]].split("\\"))[1] )
             st.image(ok)
          with col4:
            v1=(product['DESCRIPTION'][ref.index(ls[1])]).split(" ")[0:3]
            st.title((' '.join([str(elem) for elem in v1])).upper())
            st.markdown(product['PAGE URL'][ref.index(ls[1])], unsafe_allow_html=True)
        
        with tab3:
          col5, col6= st.columns(2)
          with col5:
             ok=os.path.join('https://m.media-amazon.com/images/I/', list(filenames[indices[0][2]].split("\\"))[1] )
             st.image(ok)
          with col6:
            v2=(product['DESCRIPTION'][ref.index(ls[2])]).split(" ")[0:3]
            st.title((' '.join([str(elem) for elem in v2])).upper())
            st.markdown(product['PAGE URL'][ref.index(ls[2])], unsafe_allow_html=True)
          
        with tab4:
          col7, col8= st.columns(2)
          with col7:
             ok=os.path.join('https://m.media-amazon.com/images/I/', list(filenames[indices[0][3]].split("\\"))[1] )
             st.image(ok)
          with col8:
            v3=(product['DESCRIPTION'][ref.index(ls[3])]).split(" ")[0:3]
            st.title((' '.join([str(elem) for elem in v3])).upper())
            st.markdown(product['PAGE URL'][ref.index(ls[3])], unsafe_allow_html=True)
      
        with tab5:
          col9, col10= st.columns(2)
          with col9:
             ok=os.path.join('https://m.media-amazon.com/images/I/', list(filenames[indices[0][4]].split("\\"))[1] )
             st.image(ok)
          with col10:
            v4=(product['DESCRIPTION'][ref.index(ls[4])]).split(" ")[0:3]
            st.text((' '.join([str(elem) for elem in v4])).upper())
            st.markdown(product['PAGE URL'][ref.index(ls[4])], unsafe_allow_html=True)
      
