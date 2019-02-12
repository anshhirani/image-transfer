#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:49:17 2019

@author: anshhirani

Citations:
    
    1) A Neural Algorithm of Artistic Style (http://arxiv.org/abs/1508.06576)
    2) https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
    3) Keras: https://keras.io
"""

## Import Modules ##

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras import backend as K
import numpy as np

## Constants for Code Dev. ##

style_path = "persistence-of-memory-salvador-deli.jpg"
content_path = "running-horses.jpg"
wn_path = "white-noise.jpg"

## Neural Image Style Transfer ##

def style_transfer(style_path, content_path, wn_path):
    # preprocess images: style, content, white noise
    style_mat = preprocess_img(style_path)
    content_mat = preprocess_img(content_path)
    wn_mat = preprocess_img(wn_path)   
    # VGG network features stored
    vgg_model = VGG16(weights = 'imagenet',
                           pooling = 'avg',
                           include_top = False)
    # content feed forward
    content_features = content_extract(vgg_model, content_mat)
    # style feed forward and storage
    style_features_dict = []
    
    '''
    TODO:
        - Feed forward on all 3 images, store relevant layers
        - Backprop to update WN image. K.gradients
        - initialize hyperparameters -> alpha, beta, learning rate, etc.
    '''
    
    
    return

## Image Preprocessing Functions ##

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_mat = image.img_to_array(img)
    img_mat = np.expand_dims(img_mat, axis=0)
    img_mat = preprocess_input(img_mat)
    return img_mat

## Content Picture Functions ##

def content_extract(model, content_mat):
    final_layer = 'block5_conv3'
    final_out_model = Model(inputs = model.input,
                            outputs = model.get_layer(final_layer).output)
    content_output = final_out_model.predict(content_mat)    
    return content_output


def content_loss(content_img, wn_img):
    loss = .5*(K.sum(K.square(wn_img - content_img)))
    return loss


## Style Picture Functions

def gram(wn_extract):
    corr_mat = K.dot(wn_extract, K.transpose(wn_extract))
    return corr_mat


def style_loss():
    
    return 




