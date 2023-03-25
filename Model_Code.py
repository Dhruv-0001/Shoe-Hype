#EXTRACTION FEATURES

#*******IMPORTING AND INSTALLING DEPENDENCIES*********

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
import pandas as pd
import csv


#*********CREATING MODEL & EXTRACTING FEATURES OF UPLOADED IMAGE***********

model = ResNet50 (weights='imagenet', include_top=False, input_shape=(224, 224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img("sample images/shoes2",target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)


#******EXTRACTING FEATURES FROM OUR IMAGES AND STORING THE RESULTS******

filenames_shoes = []
for file in os.listdir("SHOES_IMAGES"):
    filenames_shoes.append(os.path.join("SHOES_IMAGES",file))

feature_list_shoes = []
for file in tqdm(filenames_shoes):    
    feature_list_shoes.append(extract_features(file, model))
    
pickle.dump(feature_list_shoes, open('embeddings_shoes.pkl', 'wb'))
pickle.dump(filenames_shoes, open('filenames_shoes.pkl', 'wb'))


#PREDICTING RECOMMENDATIONS

#****LOADING EXTRACTED FEATURES************

feature_list_shoes = np.array(pickle.load(open('embeddings_shoes.pkl','rb')))
filenames_shoes = pickle.load(open('filenames_shoes.pkl','rb'))


#******FINDING NEAREST VECTORS FOR RECOMMENDATION*********

product = pd.read_csv("dataset_csv.csv")
data = os.listdir("SHOES_IMAGES")

neighbors = NearestNeighbors(n_neighbors=5, algorithm ='brute', metric='euclidean')
neighbors.fit(feature_list_shoes)
distances, indices = neighbors.kneighbors([normalized_result])

value = indices.tolist()[0]

#**********FETCHING PRODUCT URLS AND DESCRIPTION FROM DATASET*********

ls=[]
for i in value:
    ls.append(data[i]) 

for i in ls :
    ref=product['REFERENCE'].tolist()
    try :
        ind = ref.index(i)
        print(product['DESCRIPTION'][ind])
        print(product['PAGE URL'][ind])
    except:
        print('INFORMATION NOT AVAILABLE IN DATASET.')
