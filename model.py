from __future__ import print_function
import os
from preprocessing import listdir_nods


import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from scipy.stats import gaussian_kde

from preprocessing import *

import os

# Either or, just change syntax
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import *

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

import sys, os, re, random, math

print("TF version:", tf.__version__)

if tf.config.list_physical_devices('GPU'):
    print('GPU IS AVALIABLE')
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # frees up gpu cache if
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# generate a CNN based off of provided data directories
@tf.autograph.experimental.do_not_convert
def generate_cnn(train_dir: str, val_dir: str, test_dir: str, class_weight, input_shape=(120, 69, 1),
                 do_data_augmentation=False):
    tf.keras.backend.clear_session()
    tf.random.set_seed(1)

    datagen_kwargs = dict(rescale=1. / 255)
    dataflow_kwargs = dict(batch_size=2)

    # validation dataflow and datagen
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    # valid_datagen.standardize(X_test)
    valid_generator = valid_datagen.flow_from_directory(
        train_dir,
        shuffle=False, **dataflow_kwargs)

    # training dataflow and datagen
    # data augmentation code (data randomization) only applies to training set
    if do_data_augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2,
            **datagen_kwargs)
    else:
        train_datagen = valid_datagen

    train_generator = valid_generator

    # The Model
    model = Sequential()
    model.add(Dense(120, input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=2)  # if validation loss strictly increasing, stop training early

    batch_size = 16
    image_gen = ImageDataGenerator(rescale=1 / 255)
    train_image_gen = image_gen.flow_from_directory(train_dir, target_size=input_shape[:2], color_mode='grayscale',
                                                    batch_size=batch_size, class_mode='binary')

    val_image_gen = image_gen.flow_from_directory(val_dir, target_size=input_shape[:2], color_mode='grayscale',
                                                  batch_size=batch_size, class_mode='binary', shuffle=False)

    test_image_gen = image_gen.flow_from_directory(test_dir, target_size=input_shape[:2], color_mode='grayscale',
                                                   batch_size=batch_size, class_mode='binary', shuffle=False)

    history = model.fit(train_image_gen, epochs=10,
                        validation_data=val_image_gen, callbacks=[early_stop],
                        class_weight=class_weight)

    return model, test_image_gen, history

# Generate a CNN and plot the ROC curve
def generate_roc_CNN(train_dir: str, valid_dir : str, test_dir: str, modsave_path: str):
    tpr_values = []
    fpr_values = []
    roc_scores = []

    p = listdir_nods(train_dir)

    #size = np.asarray(Image.open(p[0])).shape

    # figure out relative proportions of each class and calculate the appropriate bias
    classcount = []
    for c in listdir_nods(train_dir):
        p = os.path.join(train_dir, c)
        classcount.append(len(listdir_nods(p)))
    s = sum(classcount)
    biases = []
    for count in classcount:
        b = (1 / count) * (s / 2)
        biases.append(b)

    class_weight = {0: biases[1], 1: biases[0]}

    for i in range(5):

        mod, test_gen, history = generate_cnn(train_dir=train_dir, val_dir=valid_dir, test_dir=test_dir, class_weight=class_weight)

        #print(type(mod))
        y_binary = test_gen.classes

        ### Model Metrics
        scores = mod.predict_proba(test_gen, verbose=1)
        #print(scores)
        #print(len(np.unique(scores)))
        #print(len(scores))
        #print(len(y_binary))

        preds = []
        for j in range(len(scores)):
            if scores[j] > 0.5:
                preds.append(0)
                continue
            preds.append(1)
        print(accuracy_score(y_true=y_binary, y_pred=preds))
        roc_values = []
        for thresh in np.linspace(0, 1, 100):
            preds = []
            for k in range(len(scores)):
                if scores[k] > thresh:
                    preds.append(1)
                    continue
                preds.append(0)
            #print(thresh)
            #print(confusion_matrix(y_binary, preds))
            preds = np.asarray(preds)
            #print(y_binary)
            #print(preds)
            tn, fp, fn, tp = confusion_matrix(y_binary, preds).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            roc_values.append([tpr, fpr])
        tpr_value, fpr_value = zip(*roc_values)

        tpr_values.append(tpr_value)
        fpr_values.append(fpr_value)
        roc = roc_auc_score(y_binary, scores)
        if not roc_scores:
            best = mod
        elif roc > max(roc_scores):
            best = mod
        roc_scores.append(roc)

    # pal1 = sns.color_palette("rainbow", 12)
    # sns.set_palette(pal1)

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(fpr_values[0], tpr_values[0])
    ax.plot(fpr_values[1], tpr_values[1])
    ax.plot(fpr_values[2], tpr_values[2])
    ax.plot(fpr_values[3], tpr_values[3])
    ax.plot(fpr_values[4], tpr_values[4])
    ax.plot(np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            label='baseline',
            linestyle='--', color='#000000')
    plt.ylabel('TPR', fontsize=16)
    plt.xlabel('FPR', fontsize=16)
    plt.legend(fontsize=12)
    plt.title("ROC AUC:" + str(np.mean(roc_scores)))
    plt.show()

    tf.saved_model.save(best, modsave_path)
