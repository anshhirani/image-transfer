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
    5) 5) https://github.com/hunter-heidenreich/ML-Open-Source-Implementations/blob/master/Style-Transfer/Style%20Transfer.ipynb

"""

### Import Modules ###

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from PIL import Image
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


### Preprocess Image ###

def preprocess_img(img_path, target_size):
    '''
    Preprocess image and return a tensor representation of it
    Returns preprocessed image in BGR format
    '''
    img = image.load_img(img_path, target_size = target_size)
    img_mat = image.img_to_array(img)
    img_mat = np.expand_dims(img_mat, axis=0)
    img_mat = preprocess_input(img_mat)
    return K.variable(img_mat)


### Feature Extraction ###

def extract_layers(content_matrix, style_matrix, generated_matrix):
    '''
    Runs matrices through VGG models and returns layer values
    for transfer
    '''
    input_tensor = K.concatenate([content_matrix, style_matrix, generated_matrix], axis = 0)
    vgg_model = VGG16(input_tensor = input_tensor,
                      weights = 'imagenet',
                      pooling = 'avg',
                      include_top = False)
    # input and output layers of whole model stored in dictionary
    layers_dict = dict([(layer.name, layer.output) for layer in vgg_model.layers])
    # specify layers for content and style
    content_layer = 'block4_conv2'
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']

    content_extract = layers_dict[content_layer]
    style_extracts = [layers_dict[layer] for layer in style_layers]
    return content_extract, style_extracts


### Loss Functions ###

def content_loss(content_features, generated_features):
    c_loss = .5*(K.sum(K.square(generated_features - content_features)))
    return c_loss


def gram_matrix(features):
    corr_mat = K.dot(features, K.transpose(features))
    return corr_mat


def style_loss(style_feature, generated_feature, final_width, final_height):
    '''
    For this implementation, we assume equal weights for
    weighing the layered losses.
    '''
    
    # since preprocessing matrix puts in BGR form, we permute to put in RBG form
    style_reshape = K.batch_flatten(K.permute_dimensions(style_feature, (2, 0, 1)))
    generated_reshape = K.batch_flatten(K.permute_dimensions(generated_feature, (2, 0, 1)))
    
    # get gram matrices
    style_gram = gram_matrix(style_reshape)
    generated_gram = gram_matrix(generated_reshape)
    
    s_loss = .25 * K.sum(K.square(generated_gram - style_gram)) / (3**2 * (final_width*final_height)**2)
    return s_loss


def total_loss(content_matrix, style_matrix, generated_matrix, alpha, beta, final_width, final_height):
    c_layer_extract, s_layers_extract = extract_layers(content_matrix, style_matrix, generated_matrix)
    # content loss
    content_feature = c_layer_extract[0, :, :, :]
    gen_content_feature = c_layer_extract[2, :, :, :]
    c_loss = content_loss(content_feature, gen_content_feature)
    # style loss
    s_loss = 0
    for i in range(len(s_layers_extract)):
        s_layer_extract = s_layers_extract[i]
        style_feature = s_layer_extract[1, :, :, :]
        gen_style_feature = s_layer_extract[2, :, :, :]
        s_loss += (style_loss(style_feature, gen_style_feature, final_width, final_height) / len(s_layers_extract))
    return alpha*c_loss + beta*s_loss


def evaluate_loss(generated_img, final_width, final_height, output_fn):
    '''
    Calculates differentiable loss value
    '''
    
    generated_img = generated_img.reshape((1, final_height, final_width, 3))
    outputs = output_fn([generated_img])
    loss_val = outputs[0]
    return loss_val

def evaluate_gradient(generated_img, final_width, final_height, output_fn):
    '''
    Calculates the gradient for updating
    '''
    
    generated_img = generated_img.reshape((1, final_height, final_width, 3))
    outputs = output_fn([generated_img])
    gradient_values = outputs[1].flatten().astype('float64')
    return gradient_values

### Style transfer ###

def style_transfer(style_img_path, content_img_path, gen_im_name, final_height = 256,
                   final_width = 256, alpha = 1, beta = 10, iterations = 350):
    target_size = (final_height, final_width)
    # Generate White Noise
    generated_matrix = np.random.uniform(0, 255, (1, final_height, final_width, 3))
    generated_matrix = preprocess_input(generated_matrix)
    generated_image = K.placeholder((1, final_height, final_width, 3))
    # load pictures + use the wn image made in the global scape
    content_matrix = preprocess_img(content_img_path, target_size)
    style_matrix = preprocess_img(style_img_path, target_size)
    # extract all layers
    content_layer, style_layers = extract_layers(content_matrix, style_matrix, generated_image)
    # calculate loss
    loss = total_loss(content_matrix, style_matrix, generated_image, alpha, beta, final_width, final_height)
    gradients = K.gradients(loss, generated_image)
    outputs = [loss] + gradients
    output_fn = K.function([generated_image], outputs)

    new_gen_img, new_loss_val, info = fmin_l_bfgs_b(evaluate_loss,
                                                    generated_matrix.flatten(),
                                                    fprime = evaluate_gradient,
                                                    args = (final_width, final_height, output_fn,),
                                                    maxiter = iterations)
    # save image
    save_img = new_gen_img.reshape((final_height, final_width, 3))
    save_img = save_img[:, :, ::-1]
    save_img[:, :, 0] += 103.939
    save_img[:, :, 1] += 116.779
    save_img[:, :, 2] += 123.68
    save_img = np.clip(save_img, 0, 255).astype('uint8')
    save_img = Image.fromarray(save_img)
    save_img = save_img.resize(target_size)
    save_img.save(gen_im_name)
    return