#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:49:17 2019

@author: anshhirani

Citations:
    
    1) A Neural Algorithm of Artistic Style (http://arxiv.org/abs/1508.06576)
    2) https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
    3) Keras: https://keras.io
    4) https://github.com/walid0925/AI_Artistry/blob/master/main.py

"""

## Import Modules ##

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras import backend as K
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

## Constants for Code Dev. ##

style_path = "persistence-of-memory-salvador-deli.jpg"
content_path = "running-horses.jpg"
wn_path = "white-noise.jpg"

tf_session = K.get_session()

## Neural Image Style Transfer ##

def style_transfer(style_path, content_path, wn_path, iterations = 100):
    # preprocess images: style, content, white noise
    style_mat = preprocess_img(style_path)
    content_mat = preprocess_img(content_path)
    wn_mat = preprocess_img(wn_path)   
    # VGG network features stored
    vgg_model = VGG16(weights = 'imagenet',
                           pooling = 'avg',
                           include_top = False)
    # Extractions
    style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3',
                    'block4_conv3', 'block5_conv3']
    style_extracts = extract(vgg_model, style_layers, style_mat)
    
    content_layers = ['block5_conv3']
    content_extract = extract(vgg_model, content_layers, content_mat)[0]
    
    weights = np.ones(len(style_layers))/float(len(style_layers))
    
    # Feed to Network
    input_wn = wn_mat.flatten()
    
    
    
    '''
    TODO:
        - Backprop to update WN image. K.gradients
        - Iterate & run
    '''
    
    
    return

## Image Preprocessing Functions ##

def preprocess_img(img_path):
    '''
    Preprocess image and return a tensor representation of it
    '''
    img = image.load_img(img_path, target_size=(224, 224))
    img_mat = image.img_to_array(img)
    img_mat = np.expand_dims(img_mat, axis=0)
    img_mat = preprocess_input(img_mat)
    return img_mat

## Feature Extraction ##

def extract(model, layers, image_mat):
    '''
    Get layered features in list form
    '''
    stored_features = []
    for layer in layers:
        layer_model = Model(inputs = model.input, 
                            outputs = model.get_layer(layer).output)
        layer_out = layer_model.predict(image_mat)
        # get shapes of extraction -> should be (1 x L x W x D)
        shape = K.shape(layer_out).eval(session = tf_session)
        N_l = shape[3]
        M_l = shape[1]*shape[2]
        # format final matrix for proper shape
        feature_mat = K.transpose(K.reshape(layer_out, (M_l, N_l)))
        stored_features.append(feature_mat)
    return stored_features

## Loss Functions ##

def content_loss(content_extract, wn_extract):
    loss = .5*(K.sum(K.square(wn_extract - content_extract)))
    return loss

def gram(extract):
    corr_mat = K.dot(extract, K.transpose(extract))
    return corr_mat

def style_loss(weights, wn_extracts, style_extracts):
    '''
    weights, number of gram matrices, and layers
    in the style_extracts should all be the same length
    '''
    s_loss = K.variable(0)
    for i in range(len(weights)):
        # initialize values
        weight = weights[i]
        layered_wn_extract = wn_extracts[i]
        layered_style_extract = style_extracts[i]
        # calculate layer loss
        layered_Nl = K.shape(layered_wn_extract).eval(session = tf_session)[0]
        layered_Ml = K.shape(layered_wn_extract).eval(session = tf_session)[1]
        wn_gram = gram(layered_wn_extract)
        style_gram = gram(layered_style_extract)
        layer_loss = weight*.25*K.sum(K.square(wn_gram - style_gram)) / (layered_Nl**2 * layered_Ml**2)
        s_loss += layer_loss
    return s_loss

def total_loss(model, weights, wn_mat, content_layers, style_layers,
               content_extract, style_extracts, alpha = 1, beta = 10000):
    '''
    Calculates total loss, with weights alpha and beta. 
    Alpha weights the content loss, Beta the style loss. 
    Defaults are recommended values from the paper.
    '''
    wn_content_extract = extract(model, content_layers, wn_mat)[0]
    wn_style_extracts = extract(model, style_layers, wn_mat)
    c_loss = content_loss(content_extract, wn_content_extract)
    s_loss = style_loss(weights, wn_style_extracts, style_extracts)
    total_loss_ = alpha*c_loss + beta*s_loss
    return total_loss_

def differentiable_loss(model, weights, wn_mat, content_layers, style_layers,
                        content_extract, style_extracts):
    '''
    convert loss function to a Keras-type function
    to take gradients
    '''
    if wn_mat.shape != (1, 224, 224, 3):
        wn_mat = wn_mat.reshape(1, 224, 224, 3)
    differentiable_fn = K.function([model.input],
                                   [total_loss(model,
                                               weights,
                                               model.input,
                                               content_layers,
                                               style_layers,
                                               content_extract,
                                               style_extracts)])  
    return differentiable_fn([wn_mat])[0]

def get_gradient(model, weights, wn_mat, content_layers, style_layers,
                 content_extract, style_extracts):
    '''
    Calculate gradients for differentiable loss
    '''
    if wn_mat.shape != (1, 224, 224, 3):
        wn_mat = wn_mat.reshape(1, 224, 224, 3)
    grad_fn = K.function([model.input],
                         K.gradients(total_loss(model,
                                                weights,
                                                model.input,
                                                content_layers,
                                                style_layers,
                                                content_extract,
                                                style_extracts),
                                     [model.input]))
    gradient = grad_fn([wn_mat])[0].flatten()
    return gradient






