from sklearn.model_selection import train_test_split
# https://stackoverflow.com/questions/30667525/importerror-no-module-named-sklearn-cross-validation
# sklearn.cross_validation 已被淘汰不支援
# from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras import backend as K

import random
import keras
import numpy as np

# train dataset
train_images = None # x
train_labels = None # y

#valid dataset
valid_images = None # x
valid_labels = None # y

#test dataset
test_images = None # x
test_labels = None # y

def process_data(images,labels,img_rows,img_cols ,img_channels,n_class):

    # split dataset 'train'、'valid'、'test'
    train_images,valid_images,train_labels,valid_labels = train_test_split(images,labels,test_size = 0.2, random_state = random.randint(0,n_class))
    _,test_images,_,test_labels = train_test_split(train_images,train_labels,test_size = 0.3, random_state = random.randint(0,n_class))
    
    # https://faroit.github.io/keras-docs/1.2.2/backend/

    if K.image_dim_ordering() == 'th': # img_channels,img_rows,img_cols

        train_images = train_images.reshape(train_images.shape[0],img_channels,img_rows,img_cols)
        valid_images = valid_images.reshape(valid_images.shape[0],img_channels,img_rows,img_cols)
        test_images = test_images.reshape(test_images.shape[0],img_channels,img_rows,img_cols)

    else: # img_rows,img_cols,img_channels

        train_images = train_images.reshape(train_images.shape[0],img_rows,img_cols,img_channels)
        valid_images = valid_images.reshape(valid_images.shape[0],img_rows,img_cols,img_channels)
        test_images = test_images.reshape(test_images.shape[0],img_rows,img_cols,img_channels)
    
    # normalize data
    train_images,valid_images,test_images = normalize_data(train_images,valid_images,test_images)

    # show "train_images"、"valid_imagess"、"test_images" length
    print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Train Sample: {}".format(train_images.shape[0]))
    print("Valid Sample: {}".format(valid_images.shape[0]))
    print("Test Sample: {}".format(test_images.shape[0]))
    
    #print("{}\n".format(train_images))
    #print("{}\n".format(valid_images))
    #print("{}\n".format(test_images))

    #https://blog.csdn.net/roger_royer/article/details/79968523
    # One Hot Code process "train_labels","valid_labels","test_labels"
    train_labels,valid_labels,test_labels = one_hot_process(train_labels,valid_labels,test_labels,n_class)
    print("Train Labels Sample: {}".format(train_labels.shape[0]))
    print("Valid Labels Sample: {}".format(valid_labels.shape[0]))
    print("Test Labels Sample: {}".format(test_labels.shape[0]))
    
    #print("{}\n".format(train_labels))
    #print("{}\n".format(valid_labels))
    #print("{}\n".format(test_labels))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    return train_images,valid_images,test_images,train_labels,valid_labels,test_labels

#One Hot Code for label vector
def one_hot_process(train_labels,valid_labels,test_labels,n_class):

    train_labels = np_utils.to_categorical(train_labels,n_class)
    valid_labels = np_utils.to_categorical(valid_labels,n_class)
    test_labels = np_utils.to_categorical(test_labels,n_class)

    return train_labels,valid_labels,test_labels


# normalize data
def normalize_data(train_images,valid_images,test_images):

    train_images = train_images.astype('float16')
    valid_images = valid_images.astype('float16')
    test_images = test_images.astype('float16')

    train_images /= 255
    valid_images /= 255
    test_images /= 255

    return train_images,valid_images,test_images
    