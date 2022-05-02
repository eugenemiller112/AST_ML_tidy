from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from preprocessing import *

# Either or, just change syntax
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

from sklearn.metrics import accuracy_score

import os

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
def generate_ROC_CNN(train_dir: str, valid_dir: str, test_dir: str, modsave_path: str, cross_validations: int,
                     verbose=False):
    tpr_values = []
    fpr_values = []
    roc_scores = []

    # size = np.asarray(Image.open(p[0])).shape

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

    for i in range(cross_validations):

        mod, test_gen, history = generate_cnn(train_dir=train_dir, val_dir=valid_dir, test_dir=test_dir,
                                              class_weight=class_weight)

        y_binary = test_gen.classes  # get ground truth

        # Model Metrics
        scores = mod.predict_proba(test_gen, verbose=1)
        if verbose:
            print("Scores")
            print(scores)
            print(len(np.unique(scores)))
            print(len(scores))
            print(len(y_binary))

        preds = []
        for j in range(len(scores)):

            if scores[j] > 0.5:
                preds.append(0)
                continue

            preds.append(1)

        if verbose:
            print("Accuracy score")
            print(accuracy_score(y_true=y_binary, y_pred=preds))

        roc_values = []
        for thresh in np.linspace(0, 1, 100):
            preds = []

            for k in range(len(scores)):

                if scores[k] > thresh:
                    preds.append(1)
                    continue

                preds.append(0)

            if verbose:
                print(thresh)
                print(confusion_matrix(y_binary, preds))
                print(y_binary)
                print(preds)

            preds = np.asarray(preds)
            tn, fp, fn, tp = confusion_matrix(y_binary, preds).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            roc_values.append([tpr, fpr])
        tpr_value, fpr_value = zip(*roc_values)

        tpr_values.append(tpr_value)
        fpr_values.append(fpr_value)
        roc = roc_auc_score(y_binary, scores)  # calculate ROC AUC
        if not roc_scores:
            best = mod
        elif roc > max(roc_scores):
            best = mod
        roc_scores.append(roc)
        if verbose:
            print(roc_scores)

    fig, ax = plt.subplots(figsize=(5, 7))
    for j in range(cross_validations):
        ax.plot(fpr_values[j], tpr_values[j])
    ax.plot(np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            label='baseline',
            linestyle='--', color='#000000')
    plt.ylabel('TPR', fontsize=16)
    plt.xlabel('FPR', fontsize=16)
    plt.legend(fontsize=12)
    plt.title("ROC AUC:" + str(np.max(roc_scores)))
    plt.show()

    tf.saved_model.save(best, modsave_path)


def generate_RNN(train_dir: str, val_dir: str, test_dir: str, input_shape=(120, 69, 1),
                 do_data_augmentation=False):
    # array to save training data
    train_dat = []
    train_lab = []

    # array to save validation data
    val_dat = []
    val_lab = []

    # array to save test data
    test_dat = []
    test_lab = []

    # arrays to calculate class weights
    len_data = []
    labs = []

    # looping through the different categories (Resistant, Susceptible)
    lab = 0
    # TODO check dimensions -  do I want to transpose the data?
    for cat in listdir_nods(train_dir):
        print(cat, "label: ", lab)

        # Get data to be used to calculate class weights
        len_data.append(len(listdir_nods(os.path.join(train_dir, cat))))
        labs.append(lab)

        # Get the training data for this category
        cat_train_dat, cat_train_lab = images_to_tensors(os.path.join(train_dir, cat),
                                                         xdim=input_shape[0], ydim=input_shape[1], label=lab)
        train_dat.extend(cat_train_lab)
        train_lab.extend(cat_train_lab)

        # Get the validation data for this category
        cat_val_dat, cat_val_lab = images_to_tensors(os.path.join(val_dir, cat),
                                                     xdim=input_shape[0], ydim=input_shape[1], label=lab)
        val_dat.extend(cat_val_lab)
        val_lab.extend(cat_val_lab)

        # Get the test data for this category
        cat_test_dat, cat_test_lab = images_to_tensors(os.path.join(test_dir, cat),
                                                       xdim=input_shape[0], ydim=input_shape[1], label=lab)
        test_dat.extend(cat_test_lab)
        test_lab.extend(cat_test_lab)

        lab += 1

    # Calculate the class weights and save to dict.
    class_weight = dict()
    n_data = np.sum(len_data)
    for l in labs:
        print(len_data[l])
        class_weight[l] = len_data[l] / n_data

    # The machine learns!!
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(256, return_sequences=True))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(layers.SimpleRNN(128))

    model.add(layers.Dense(2, activation='sigmoid'))

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=2)  # if validation loss strictly increasing, stop training early

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer="SGD",
                  metrics=["accuracy"])

    history = model.fit(x=train_dat, y=train_lab, epochs=10,
                        validation_data=(val_dat, val_lab), callbacks=[early_stop],
                        class_weight=class_weight)

    return model, test_dat, test_lab, history
