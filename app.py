# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:18:03 2022

@author: planz
"""


#------------------------------------------------------------------------
#   LIBRARIES
#------------------------------------------------------------------------

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import streamlit as st
from streamlit_cropper import st_cropper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageOps
from pywaffle import Waffle
from tensorflow import keras
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image, ImageOps

import tensorflow.keras.backend as K
import tensorflow as tf
import cv2
import mediapipe as mp


#------------------------------------------------------------------------
#   GLOBALS
#------------------------------------------------------------------------

model_loc = os.getcwd() + "\\"
model_name = 'resnet50_best.h5'
path = model_loc + "Pictures\\"
weights_file = model_loc + model_name  

# Load metadata
df = pd.read_csv(model_loc + 'metadata_best.csv')
labs = list(df['labs'])
names = list(df['cats'])
cols = list(df['colors'])

width1 = 350
width2 = 350
width3 = 240
st.set_option('deprecation.showPyplotGlobalUse', False)



#------------------------------------------------------------------------
#   FUNCTIONS
#------------------------------------------------------------------------

def load_image(image_file):
	img = Image.open(image_file)
	return img


def label_adjust(data, labels):
    '''
    Function to remove chart labels for low confidence categories
    '''
    for i in range(len(data)):
        if data[i] < 0.05:
            labels[i] = ''
    
    return labels
    


def donut(data, labels, cols):
    '''
    Function to generate a donut chart based on prediction confidence
    '''
    # create data
    size_of_groups = data.reshape(data.shape[1])
    
    # Clean up the labels
    items = label_adjust(size_of_groups, names)
    
    # Create a pieplot
    plt.pie(size_of_groups, labels=items, colors=cols, labeldistance=1.15)
    plt.title('Prediction Confidence', fontsize=16)
    # add a circle at the center to transform it in a donut chart
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    
    plt.show()
        

def waffle(data, labels, cols):
            
    fig = plt.figure(FigureClass=Waffle, rows=5, 
    values = data.reshape(data.shape[1])*100, 
    colors = cols,
    labels = labels,
    legend = {'loc':'center', 'fontsize':12, 'bbox_to_anchor': (0.5, -1),
        'ncol': len(data),
        'framealpha': 0}
    )
    plt.title('Prediction Confidence', fontsize=16)
    plt.show()



def gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    '''
    Original function author:  Chollet, F (2020)
    https://keras.io/examples/vision/grad_cam/
    '''
    # Convert image to array
    img_array = np.expand_dims(img, axis=0)
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()



def display_gradcam(img, heatmap, alpha=0.8):
    '''
    Original function author:  Chollet, F (2020)
    https://keras.io/examples/vision/grad_cam/
    '''
        
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    img = np.array(img)
    
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    #superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))
    
    return superimposed_img


def background(image, slider_value):

    bg_image = np.ones((10,10,3)) * slider_value * 25.5
    
    height, width, channels = image.shape
    
    # initialize mediapipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # get the result
    results = selfie_segmentation.process(RGB)
    
    # extract segmented mask
    #mask = results.segmentation_mask
    
    condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.5
    
    
    # resize the background image to the same size of the original frame
    bg_image = cv2.resize(bg_image, (width, height))
    
    # combine frame and background image using the condition
    output_image = np.where(condition, image, bg_image)
    
    return output_image


    
    


#------------------------------------------------------------------------
#   DISPLAY
#------------------------------------------------------------------------

# Set the width of the sidebar and main screen
pg_icon = Image.open(path + "black_bin.jpg")
st.set_page_config(layout="wide", page_title="Recycling Classifier", page_icon=pg_icon)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 220px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 220px;
        margin-left: -220px;
    }
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 1rem;}     
    """,
    unsafe_allow_html=True,
    )



# Screen set up
st.title("Recycling Classifier")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
my_button = st.sidebar.radio("Confidence Graph Type:", ('Donut Chart', 'Waffle Chart')) 
grad = st.sidebar.checkbox(label="Prediction Heatmap", value=True)
crop = st.sidebar.checkbox(label="Crop Image", value=False)
bgd = st.sidebar.checkbox(label="Background", value=False)
col1, col2, col3 = st.columns(3)



#------------------------------------------------------------------------
#   MAIN
#------------------------------------------------------------------------

# Collect image and infer class
if image_file is not None:

    # To See details
    file_details = {"filename":image_file.name, "filetype":image_file.type, 
                    "filesize":image_file.size}
    
    # To View Uploaded Image
    col1.image(load_image(image_file),width=width1)
    
    # Generate the prediction
    model = keras.models.load_model(weights_file)
        
    image = Image.open(image_file)
    
    # Check for image cropping
    if crop == True:
        image = ImageOps.fit(image, (350,263), Image.ANTIALIAS)
        image = st_cropper(image, realtime_update=crop, aspect_ratio=None)
        
    
    size = (224,224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    im = np.array(image)
    
    # Background noise reduction
    if bgd == True:
        slider = st.sidebar.slider(label="", min_value=0, 
                                   max_value=10, value=5, step=1, disabled=False)    
      
        # Pass to background function
        im = background(im, slider)
        col1.image(im/255., width=width3)
        image = im  # for Grad-Cam
    
    x = np.expand_dims(im, axis=0)
    pred_res = model.predict(x)
    class_label = labs[np.argmax(pred_res)]
    conf = np.round(np.max(pred_res)*100, 1)
    
    
    # Calculate confidence ratio
    temp = list(pred_res.reshape(5,).copy())
    temp.sort()
    conf_ratio = conf/(100*temp[-2])
    
    # Define chart type
    if my_button == 'Donut Chart':
        fig = donut(pred_res, labs, cols)
    else:
        fig = waffle(pred_res, names, cols)    
    
    # Path to location of icons
    st.markdown("""<style>.mid-font {font-size:20px !important;}</style>""", 
                    unsafe_allow_html=True)
    st.markdown("""<style>.big-font {font-size:25px !important;}</style>""", 
                    unsafe_allow_html=True)
    
    # Test the confidence level and report low confidence    
    if conf_ratio < 1.5 or conf < 60:
        st.warning('LOW CONFIDENCE PREDICTION  -  ' + str(conf) + '%')
        st.markdown('<p class="mid-font">Try cropping the image or taking another photograph of the item.</p>', unsafe_allow_html=True)        
    else:
        st.success('PREDICTION CONFIDENCE -  ' + str(conf) + '%')
        
    # Display icons
    if class_label == 'yellow_bin':
        st.markdown('<p class="big-font">PLASTIC, GLASS, ALUMINIUM & STEEL</p>', unsafe_allow_html=True)
        st.markdown('<p class="mid-font">Place in the YELLOW bin</p>', unsafe_allow_html=True)        
        col2.image(load_image(path + 'yellow_bin.jpg'),width=width2)
        col3.pyplot(fig)
        
    elif class_label == 'blue_bin':
        st.markdown('<p class="big-font">PAPER, CARDBOARD & CARTONS</p>', unsafe_allow_html=True)
        st.markdown('<p class="mid-font">Place in the BLUE bin</p>', unsafe_allow_html=True)        
        col2.image(load_image(path + 'blue_bin.jpg'),width=width2)
        col3.pyplot(fig)
        
    elif class_label == 'trash':
        st.markdown('<p class="big-font">LANDFILL</p>', unsafe_allow_html=True)
        st.markdown('<p class="mid-font">Place in the RED bin</p>', unsafe_allow_html=True)
        col2.image(load_image(path + 'red_bin.jpg'),width=width2)
        col3.pyplot(fig)
        
    elif class_label == 'softplastic':
        st.markdown('<p class="big-font">SOFT PLASTIC</p>', unsafe_allow_html=True)
        st.markdown('<p class="mid-font">Deposit at the REDCycle bin - Coles Lane Cove, The Canopy.</p>', unsafe_allow_html=True)
        st.markdown('<p class="mid-font">Do NOT place in the red, blue or yellow bins.</p>', unsafe_allow_html=True)    
        col2.image(load_image(path + 'black_bin.jpg'),width=width2)
        col3.pyplot(fig)
        
    else:
        st.markdown('<p class="big-font">ELECTRONIC & SPECIAL WASTE</p>', unsafe_allow_html=True) 
        st.markdown('<p class="mid-font">Deposit at Community Recycling Centre -  8 Waltham St, Artarmon.</p>', unsafe_allow_html=True)
        st.markdown('<p class="mid-font">Electronic waste can be left at the E-Waste bins at The Canopy, Lane Cove.</p>', unsafe_allow_html=True)
        st.markdown('<p class="mid-font">Do NOT place in the red, blue or yellow bins.</p>', unsafe_allow_html=True)    
        col2.image(load_image(path + 'cyan_bin.jpg'),width=width2)
        col3.pyplot(fig)
        
    # Grad Cam heatmap    
    if grad == True:
        heat = gradcam_heatmap(image, model, "conv5_block3_out", pred_index=None)
        grad = display_gradcam(image, heat, alpha=0.9)
        col1.image(grad,width=width3)
    
   
