# Implementation of AlexNet model taken from
# https://www.mydatahack.com/building-alexnet-with-keras/
# script that reads data, creates model and trains it
















from tensorflow import keras as tf
import tensorflow
















from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D
















from tensorflow.keras.layers import BatchNormalization
import numpy as np
import directory_structure
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from collections import deque
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models


global model




# classes model needs to learn to classify
CLASSES_TO_CHECK = ['L', 'N', 'V', 'A', 'R']
NUMBER_OF_CLASSES = len(CLASSES_TO_CHECK)
IMAGES_TO_TRAIN = 2450
















# removing warning for tensorflow about AVX support
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
































def saveMetricsAndWeights(score, model):
    '''
    Save metric and weight data in specific folders
















    Args:
        score (list): list containing loss value and accuracy of current model
















        model (keras model): compiled model which has been trained with training data
        Returns:
                (dataframe): dataframe contatining image information
        '''
    loss = score[0]
    current_acc = score[1]
    base_path='/content/drive/MyDrive/testing/'








    directory_structure.getWriteDirectory('testing', None)
    weights_path =  base_path +'model_weights/'
    metrics_path = base_path + 'accuracy_metrics/'
    print("weights_path :"+weights_path)
    print("metrics_path :"+metrics_path)
















    if (len(directory_structure.filesInDirectory('.npy', metrics_path)) == 0):
        # create text file with placeholder accuracy value (i.e 0)
        np.save(metrics_path + 'metrics.npy', [0])
        model.save(weights_path + 'my_model.h5')
        del model
    else:
        highest_acc = np.load(metrics_path + 'metrics.npy')[0]
        if (current_acc > highest_acc):
            np.save(metrics_path + 'metrics.npy', [current_acc])
            model.save(weights_path + 'my_model.h5')
            del model
            print('\nAccuracy Increase: ' +
                  str((current_acc - highest_acc)*100) + '%')
























    # Load the .npy file
    metric_data = np.load(metrics_path +'metrics.npy')








    # Inspect the content
    print(metric_data)        
























































def getSignalDataFrame():
    '''
    Read signal images present in the directory `beat_write_dir`
    and save them in a DataFrame.
















    Returns:
        (dataframe): DataFrame containing image information
    '''
    # Get paths for where signals are present
    signal_path = directory_structure.getWriteDirectory('beat_write_dir', None)
    signal_path = '/content/drive/MyDrive/beat_write_dir/'
















    # Create DataFrame
    df = pd.DataFrame(columns=['Signal ID', 'Signal', 'Type'])
















    arrhythmia_classes = directory_structure.getAllSubfoldersOfFolder(signal_path)
    print("getSignalDataFrame arrhythmia_classes:", arrhythmia_classes)
    print("1 df head:", df.head())
















    image_paths = deque()
    image_ids = deque()
    class_types = deque()
    images = []
















    # Get path for each image in classification folders
    for classification in arrhythmia_classes:
        print("classification:", classification)
        classification_path = ''.join([signal_path, classification])
        image_list = directory_structure.filesInDirectory('.png', classification_path)
















        # Initialize your counter 'i'
        i = 0
















        # Loop through the image_list
        for beat_id in image_list:
          print("beat_id:", beat_id)
















          # Increment 'i'
          i += 1
















          # Check if the condition is met (I assume you want to use 'i < 50')
          if i < 4:
            # Append relevant data to the lists
            print("beat_id:", beat_id)
            image_ids.append(directory_structure.removeFileExtension(beat_id))
            class_types.append(classification)
            image_paths.append(''.join([classification_path, '/', beat_id]))
          else:
            # Reset 'i' and continue
            i = 0
            break
               
















    # Read and save images in the DataFrame
    for path in image_paths:
        print("path:", path)
        images.append(cv2.imread(path))
















    # Save information in the DataFrame
    df['Signal ID'] = list(image_ids)
    df['Type'] = list(class_types)
    df['Signal'] = images
    print("2 df head:", df.head())
















    return df
















































def normalizeData(X_train, X_test, y_train, y_test):
    '''
    Normalizing the test and train data
    '''
















    # image normalization
    X_train = X_train.astype('float32')
    X_train = X_train / 255
    X_test = X_test.astype('float32')
    X_test = X_test / 255
















    # label normalization
    y_train = keras.utils.to_categorical(y_train, NUMBER_OF_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUMBER_OF_CLASSES)
















    return X_train, X_test, y_train, y_test
































def convertToNumpy(X_train, X_test, y_train, y_test):
    '''
    Convert data arrays into numpy arrays
    '''
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
































def trainAndTestSplit(df, size_of_test_data):
    '''
        take dataframe and divide it into train and
    test data for model training
















        Args:
                df (dataframe): dataframe with all images information
















        images_to_train (int): number of images to get for training
                               from dataframe
















        size_of_test_data (float): percentage of data specified for training
















        Returns:
                X_train (list): list of training signals
















        X_test (list): list of testing signals
















        y_train (list): list of training classes
















        y_test (list): list of testing classes
        '''
    image_count = 0
    classes_to_check = CLASSES_TO_CHECK
    images_available_in_class = IMAGES_TO_TRAIN
















    # train + test data (signals and classes of signals respectively)
    X = []
    y = []
















    for index, row in df.iterrows():
        # check if current row is one of the classes to classify
        if row['Type'] in classes_to_check:
            images_available_in_class = df['Type'].value_counts()[row['Type']]
















            X.append(row['Signal'])
            y.append(classes_to_check.index(row['Type']))
            image_count += 1
















            if images_available_in_class < IMAGES_TO_TRAIN:
                if image_count == df['Type'].value_counts()[row['Type']]:
















                    image_count = 0
                    classes_to_check.remove(row['Type'])
            else:
                if image_count == IMAGES_TO_TRAIN:
















                    image_count = 0
                    classes_to_check.remove(row['Type'])
















        # if data collected from all classes break loop
        if len(classes_to_check) == 0:
            break
















    # split x and y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=size_of_test_data)
















    # convert to numpy array
    X_train, X_test, y_train, y_test = convertToNumpy(
        X_train, X_test, y_train, y_test)
















    # normalize data for easy data processing
    X_train, X_test, y_train, y_test = normalizeData(
        X_train, X_test, y_train, y_test)
















    return X_train, X_test, y_train, y_test
































def printTestMetrics(score):
    '''
    print prediction score
















    Args:
        score (list): list with test loss and test accuracy
    '''
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
































def createModel(model_name):
    '''
    Implementation of model to train images (Alexnet or Novelnet)
















    Args:
        model_name (str): name of the model to create (can choose from Alexnet and Novelnet)
















    Returns:
        model (model): model object implementation of alexnet
    '''
    base_path='/content/drive/MyDrive/testing/'
     # Check if directory exists; if not, create it
    if not os.path.exists(base_path):
        os.makedirs(base_path)








    load_from_path='/content/drive/MyDrive/testing/model_weights/'
    model_loaded = False
     # If there's a model file to load, load it and return the model
    if load_from_path and os.path.exists(load_from_path):
        print(f"Loading model from {load_from_path}")
        model = load_model(load_from_path +'my_model.h5')
        model_loaded = True
        return model


    print("1 Loading model before return", model_loaded)
    if model_loaded:
      return


    print("2 Loading model after return", model_loaded)
   
    model = Sequential()
   








    # if model_name == 'Alexnet_new':
    #   l2_lambda=1e-4
    #   input_shape=(224, 224, 3)
    #   num_classes=1000
    #    # 1st Convolutional Layer
    #   model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu',
    #                           kernel_regularizer=regularizers.l2(l2_lambda),
    #                           input_shape=input_shape))
    #   model.add(layers.BatchNormalization())
    #   model.add(layers.MaxPooling2D((3, 3), strides=2))








    #   # 2nd Convolutional Layer
    #   model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu',
    #                           kernel_regularizer=regularizers.l2(l2_lambda)))
    #   model.add(layers.BatchNormalization())
    #   model.add(layers.MaxPooling2D((3, 3), strides=2))








    #   # 3rd Convolutional Layer
    #   model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu',
    #                           kernel_regularizer=regularizers.l2(l2_lambda)))








    #   # 4th Convolutional Layer
    #   model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu',
    #                           kernel_regularizer=regularizers.l2(l2_lambda)))








    #   # 5th Convolutional Layer
    #   model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu',
    #                           kernel_regularizer=regularizers.l2(l2_lambda)))
    #   model.add(layers.MaxPooling2D((3, 3), strides=2))








    #   # Flatten and Fully Connected Layers
    #   model.add(layers.Flatten())
    #   model.add(layers.Dense(4096, activation='relu',
    #                         kernel_regularizer=regularizers.l2(l2_lambda)))
    #   model.add(layers.Dropout(0.5))
    #   model.add(layers.Dense(4096, activation='relu',
    #                         kernel_regularizer=regularizers.l2(l2_lambda)))
    #   model.add(layers.Dropout(0.5))
    #   model.add(layers.Dense(num_classes, activation='softmax',
    #                         kernel_regularizer=regularizers.l2(l2_lambda)))








    if model_name == 'Alexnet':
        # -----------------------1st Convolutional Layer--------------------------
        model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11),
                         strides=(4, 4), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())








        # -----------------------2nd Convolutional Layer---------------------------
        model.add(Conv2D(filters=256, kernel_size=(
            11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())








        # -----------------------3rd Convolutional Layer----------------------------
        model.add(Conv2D(filters=384, kernel_size=(
            3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())








        # -----------------------4th Convolutional Layer----------------------------
        model.add(Conv2D(filters=384, kernel_size=(
            3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())








        # -----------------------5th Convolutional Layer----------------------------
        model.add(Conv2D(filters=256, kernel_size=(
            3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())








        # Passing it to a dense layer
        model.add(Flatten())
        # -------------------------1st Dense Layer----------------------------
        model.add(Dense(4096, input_shape=(224*224*3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())








        # -------------------------2nd Dense Layer---------------------------
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())








        # -------------------------3rd Dense Layer---------------------------
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())








        # --------------------------Output Layer-----------------------------
        model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
















    elif model_name == 'Novelnet':
        # -----------------------1st Convolutional Layer--------------------------
        model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(7, 7),
                        strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                              strides=(2, 2), padding='same'))
        # Batch Normalisation
        model.add(BatchNormalization())


        # -----------------------2nd Convolutional Layer---------------------------
        model.add(Conv2D(filters=256, kernel_size=(5, 5),
                        strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                              strides=(2, 2), padding='same'))
        # Batch Normalisation
        model.add(BatchNormalization())


        # -----------------------3rd Convolutional Layer----------------------------
        model.add(Conv2D(filters=384, kernel_size=(3, 3),
                        strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())


        # -----------------------4th Convolutional Layer----------------------------
        model.add(Conv2D(filters=384, kernel_size=(3, 3),
                        strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())


        # -----------------------5th Convolutional Layer----------------------------
        model.add(Conv2D(filters=256, kernel_size=(3, 3),
                        strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())


        # -----------------------6th Convolutional Layer----------------------------
        model.add(Conv2D(filters=256, kernel_size=(3, 3),
                        strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                              strides=(2, 2), padding='same'))
        # Batch Normalisation
        model.add(BatchNormalization())


        # -----------------------Flatten and Dense Layers--------------------------
        model.add(Flatten())


        # -------------------------1st Dense Layer----------------------------
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())


        # -------------------------2nd Dense Layer---------------------------
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.6))
        # Batch Normalisation
        model.add(BatchNormalization())


        # -------------------------3rd Dense Layer---------------------------
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.5))
        # Batch Normalisation
        model.add(BatchNormalization())


        # --------------------------Output Layer-----------------------------
        model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))








    return model
































if __name__ == '__main__':
















    # (2) GET DATA
    df = getSignalDataFrame()
















    X_train, X_test, y_train, y_test = trainAndTestSplit(df, 0.15)
















    # (3) CREATE SEQUENTIAL MODEL
    model = createModel('Alexnet')
















   # uncomment to do computation on multiple gpus
    #parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model = model








    optimizer_sgt = tensorflow.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, weight_decay=1e-4)








    # (4) COMPILE MODEL
    parallel_model.compile(
        loss='categorical_crossentropy',
        optimizer = optimizer_sgt,
        metrics=['accuracy']
    )
















    # (5) TRAIN
    history = parallel_model.fit(
        X_train,
        y_train,
        batch_size=6,
        epochs=3,
        verbose=1,
        validation_data=(X_test, y_test),
        shuffle=True,
    )
















    # (6) PREDICTION
    predictions = parallel_model.predict(X_test)
    score = parallel_model.evaluate(X_test, y_test, verbose=0)
















    printTestMetrics(score)
















    # (7) SAVE TESTS + WEIGHTS
    saveMetricsAndWeights(score, parallel_model)









































