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

tf.config.run_functions_eagerly(True)


class CustomAdam(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="CustomAdam", **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))  # handle lr=learning_rate
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_v", beta1)
        self._set_hyper("beta_s", beta2)
        self._set_hyper("epsilon", epsilon)
        self._set_hyper("corrected_v", beta1)
        self._set_hyper("corrected_s", beta2)

    def _create_slots(self, var_list):
        """
        One slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "beta_v")
            self.add_slot(var, "beta_s")
            self.add_slot(var, "epsilon")
            self.add_slot(var, "corrected_v")
            self.add_slot(var, "corrected_s")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform an optimization step for the model variable.
        """

        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)  # handle learning rate decay

        momentum_var1 = self.get_slot(var, "beta_v")
        momentum_hyper1 = self._get_hyper("beta_v", var_dtype)

        momentum_var2 = self.get_slot(var, "beta_s")
        momentum_hyper2 = self._get_hyper("beta_s", var_dtype)

        momentum_var1.assign(momentum_var1 * momentum_hyper1 + (1. - momentum_hyper1) * grad)

        momentum_var2.assign(momentum_var2 * momentum_hyper2 + (1. - momentum_hyper2) * (grad ** 2))

        # Adam bias-corrected estimate

        corrected_v = self.get_slot(var, "corrected_v")
        corrected_v.assign(momentum_var1 / (1 - (momentum_hyper1 ** (self.iterations.numpy() + 1))))

        corrected_s = self.get_slot(var, "corrected_s")
        corrected_s.assign(momentum_var2 / (1 - (momentum_hyper2 ** (self.iterations.numpy() + 1))))

        epsilon_hyper = self._get_hyper("epsilon", var_dtype)

        var.assign_add(-lr_t * (corrected_v / (tf.sqrt(corrected_s) + epsilon_hyper)))

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "beta_v": self._serialize_hyperparameter("beta_v"),
            "beta_s": self._serialize_hyperparameter("beta_s"),
            "epsilon": self._serialize_hyperparameter("epsilon"),
        }


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

    n_train = 0
    n_val = 0
    n_test = 0
    for cat in listdir_nods(train_dir):
        n_train += len(listdir_nods(os.path.join(train_dir, cat)))
        n_val += len(listdir_nods(os.path.join(val_dir, cat)))
        n_test += len(listdir_nods(os.path.join(test_dir, cat)))

    # array to save training data
    train_dat = np.empty((n_train, input_shape[0], input_shape[1]))
    train_lab = np.empty(n_train)

    # array to save validation data
    val_dat = np.empty((n_val, input_shape[0], input_shape[1]))
    val_lab = np.empty(n_val)

    # array to save test data
    test_dat = np.empty((n_test, input_shape[0], input_shape[1]))
    test_lab = np.empty(n_test)

    # arrays to calculate class weights
    len_data = []
    len_val = []
    len_test = []
    labs = []

    # looping through the different categories (Resistant, Susceptible)
    lab = 0
    # TODO check dimensions -  do I want to transpose the data?
    for cat in listdir_nods(train_dir):
        print(cat, "label: ", lab)

        # Get data to be used to calculate class weights
        len_data.append(len(listdir_nods(os.path.join(train_dir, cat))))
        labs.append(lab)

        len_val.append(len(listdir_nods(os.path.join(val_dir, cat))))
        len_test.append(len(listdir_nods(os.path.join(test_dir, cat))))

        # Get the training data for this category
        cat_train_dat, cat_train_lab = images_to_tensors(os.path.join(train_dir, cat),
                                                         xdim=input_shape[0], ydim=input_shape[1], label=lab)

        # Get the validation data for this category
        cat_val_dat, cat_val_lab = images_to_tensors(os.path.join(val_dir, cat),
                                                     xdim=input_shape[0], ydim=input_shape[1], label=lab)

        # Get the test data for this category
        cat_test_dat, cat_test_lab = images_to_tensors(os.path.join(test_dir, cat),
                                                       xdim=input_shape[0], ydim=input_shape[1], label=lab)

        if not lab:
            train_dat[0:len_data[lab], :, :] = np.transpose(cat_train_dat, (2, 0, 1))
            train_lab[0:len_data[lab]] = cat_train_lab

            val_dat[0:len_val[lab], :, :] = np.transpose(cat_val_dat, (2, 0, 1))
            val_lab[0:len_val[lab]] = cat_val_lab

            test_dat[0:len_test[lab], :, :] = np.transpose(cat_test_dat, (2, 0, 1))
            test_lab[0:len_test[lab]] = cat_test_lab
        else:
            train_dat[len_data[lab-1]:len_data[lab-1]+len_data[lab], :, :] = np.transpose(cat_train_dat, (2, 0, 1))
            train_lab[len_data[lab-1]:len_data[lab-1]+len_data[lab]] = cat_train_lab

            val_dat[len_val[lab - 1]:len_val[lab-1]+len_val[lab], :, :] = np.transpose(cat_val_dat, (2, 0, 1))
            val_lab[len_val[lab - 1]:len_val[lab-1]+len_val[lab]] = cat_val_lab

            test_dat[len_test[lab - 1]:len_test[lab-1]+len_test[lab], :, :] = np.transpose(cat_test_dat, (2, 0, 1))
            test_lab[len_test[lab - 1]:len_test[lab-1]+len_test[lab]] = cat_test_lab

        lab += 1

    # Calculate the class weights and save to dict.
    class_weight = dict()
    n_data = np.sum(len_data)
    for l in labs:
        class_weight[l] = len_data[l] / n_data

    # The machine learns!!
    model = keras.Sequential()

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(512, input_shape=(120, 69)))

    #model.add(layers.Embedding(output_dim=3, input_dim=2))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    #model.add(layers.SimpleRNN(128))

    model.add(layers.Dense(2, activation='sigmoid'))

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=2)  # if validation loss strictly increasing, stop training early

    model.summary()

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=CustomAdam(),
                  metrics=["accuracy"])

    history = model.fit(x=train_dat, y=train_lab, epochs=10, batch_size=32,
                        validation_data=(val_dat, val_lab), callbacks=[early_stop],
                        class_weight=class_weight)

    return model, test_dat, test_lab, history
