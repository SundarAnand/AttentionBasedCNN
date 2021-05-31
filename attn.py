# tensorflow, keras
import keras
from keras.layers import merge, Conv2D, Input, Reshape, Activation
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from numpy import genfromtxt
from keras.preprocessing.image import img_to_array as img_to_array
from keras.preprocessing.image import load_img as load_img

# essentials
import numpy as np
import math
import os
import random
import pandas as pd
import tensorflow as tf
import argparse
from matplotlib import pyplot as plt
from keras import callbacks
from keras.callbacks import ModelCheckpoint

# tensorflow, keras
import keras
from keras.layers import merge, Conv2D, Input, Reshape, Activation, Add, Multiply
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from numpy import genfromtxt
from keras.preprocessing.image import img_to_array as img_to_array
from keras.preprocessing.image import load_img as load_img

# essentials
import numpy as np
import math
import os
import random
import pandas as pd
import tensorflow as tf
import argparse
# from matplotlib.pyplot import pyplot as plt
from keras import callbacks
from keras.callbacks import ModelCheckpoint

# attention needed
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Reshape
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import random


def AttentionConvNet(input_shape):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # First Block
    # CONV layer
    conv_1_out = Conv2D(32, (7, 7), strides=(1, 1), activation='relu', padding='same')(X_input)
    # MAXPOOL + BatchNorm
    X = MaxPooling2D((2, 2), strides=2, padding='same')(conv_1_out)
    X = BatchNormalization(axis=-1)(X)

    block_2_in1_conv = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(conv_1_out)
    block_2_in1_max = MaxPooling2D((2, 2), strides=1, padding='same')(block_2_in1_conv)

    # attention
    dense_from_block_1 = densor_block2(block_2_in1_max)
    activator_from_block_1 = activator(dense_from_block_1)
    dotProduct = mult([activator_from_block_1, block_2_in1_max])

    attention_input_2 = Add()([dotProduct, X])

    # Second Block
    # CONV layer
    conv_2_out = Conv2D(64, (5, 5), strides=(2, 2), activation='relu', padding='same')(attention_input_2)
    # MAXPOOL
    X = MaxPooling2D((2, 2), strides=2, padding='same')(conv_2_out)
    X = BatchNormalization(axis=-1)(X)

    block_3_in2_conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv_2_out)
    block_3_in2_max = MaxPooling2D((2, 2), strides=2, padding='same')(block_3_in2_conv)

    block_3_in1_conv = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(block_2_in1_max)
    block_3_in1_max = MaxPooling2D((2, 2), strides=2, padding='same')(block_3_in1_conv)

    # attention
    dense_from_block_1 = densor_block3(block_3_in1_max)
    activator_from_block_1 = activator(dense_from_block_1)
    dotProduct_1 = mult([activator_from_block_1, block_3_in1_max])

    # attention
    dense_from_block_2 = densor_block3(block_3_in2_max)
    activator_from_block_2 = activator(dense_from_block_2)
    dotProduct_2 = mult([activator_from_block_2, block_3_in2_max])

    attention_input_3 = Add()([dotProduct_1, dotProduct_2, X])

    # Third Block
    conv_3_out = Conv2D(128, (5, 5), strides=(1, 1), activation='relu', padding='same')(attention_input_3)
    # MAXPOOL
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv_3_out)
    X = BatchNormalization(axis=-1)(X)

    # Top layer
    X = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = Conv2D(64, (7, 7), strides=(2, 2), activation='relu')(X)

    # L2 normalization
    X = Lambda(lambda x: K.l2_normalize(x, axis=-1))(X)

    # Final output layer. First Unit is a sigmoid act(whether seen img is infected/not)
    # next 2 units for identifying type of infection if 1st element is 1. otherwise, don't care.
    typeOfInfection = Conv2D(4, (1, 1), strides=(1, 1), activation='softmax')(X)
    reshapeOut = Reshape((4,))(typeOfInfection)

    model = Model(inputs=X_input, outputs=reshapeOut)

    return model


def load_image(image_path, size=256):
    # data augmentation logic such as random rotations can be added here
    return img_to_array(load_img(image_path, target_size=(size, size, 3))) / 255.


class InfectedLeavesSequence(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X, y2, batch_size=256, dim=(256, 256), n_channels=3, shuffle=True):
        # Initialization
        self.dim = dim
        self.X = X
        self.y2 = y2
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batched_image_names = [self.X[k] for k in indexes]
        batched_y2 = self.y2[indexes]

        # Generate data
        X = self.__data_generation(batched_image_names)

        return X, batched_y2

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, images):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))

        # Generate data
        for i, ID in enumerate(images):
            # Store sample
            X[i, :, :, :] = load_image('data_images/Archive/' + ID)

        return X


job_dir = ''

random.seed(0)
from sklearn.model_selection import StratifiedKFold


def prep_data(df_path='infectedLeaves.csv', n_classes1=2, n_classes2=3):
    df = pd.read_csv(df_path)
    
    df['infectionType'] = df['infectionType'].map({0: 0, 1: 1, 2: 2, 4: 3})
    # create categorical quants
    y2 = keras.utils.to_categorical(df['infectionType'], num_classes=n_classes2 + 1)
    # drop final axis=-1 last dim of y2
    # final images list names
    X = df['pathName'].values
    return df, X, y2


df, X, y2 = prep_data()


def loss(y_true, y_pred, N=128, beta=128, epsilon=1e-8):
    infection_type = y_pred

    return tf.losses.softmax_cross_entropy(y_true, infection_type)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cvscores = []

##Setting up the path for saving logs
logs_path = job_dir + 'logs/tensorboard'

from sklearn.metrics import confusion_matrix

##Using the GPU
with tf.device('/device:GPU:0'):
    tf.reset_default_graph()

    # Defined shared layers as global variables
    concatenator = Concatenate(axis=-1)
    densor_block2 = Dense(1, activation="relu")
    densor_block3 = Dense(1, activation="relu")
    activator = Activation('softmax',
                           name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
    dotor = Dot(axes=2)
    mult = Multiply()

    ## Initializing the model
    new_model = AttentionConvNet((256, 256, 3))
    new_model.load_weights("saveAttentionWeights/attentionWeights.hdf5")
    ## Compling the model
    new_model.compile(optimizer="adam", loss=loss, metrics=["accuracy"]);

    ## Printing the modle summary
    print(new_model.summary())
    # new_model.load_weights("saveAttentionWeights/attentionWeights.hdf5")
    # checkpoint
    filepath = "saveAttentionWeights/attentionWeights.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=True, mode='auto',
                                 period=1)
    callbacks_list = [checkpoint]

    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(kfold.split(X, df['infectionType'])):

        print("Training on fold " + str(index + 1) + "/5...")
        #        print ("Evaluating on fold" + str(index+1) +"/5")
        m_train = len(train_indices)
        m_val = len(val_indices)
        X_train, X_val = X[train_indices], X[val_indices]
        y2_train, y2_val = y2[train_indices], y2[val_indices]

        train_seq = InfectedLeavesSequence(X_train, y2_train)
        val_seq = InfectedLeavesSequence(X_val, y2_val)

        # print(model.evaluate_generator(val_seq)

        try:
            hist = new_model.fit_generator(generator=train_seq,
                                           validation_data=val_seq,
                                           epochs=1,
                                           callbacks=callbacks_list
                                           )

            # list all data in history
            print(history.history.keys())
            print('accuracy', history.history['acc'])
            print('val acc', history.history['val_acc'])

            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')

            plt.show()
            plt.savefig('saveAttentionWeights/acc' + str(index + 1) + '.png')
            # summarize history for loss
            print('loss', history.history['loss'])
            print('val_loss', history.history['val_loss'])

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
            plt.savefig('saveAttentionWeights/loss' + str(index + 1) + '.png')
        except:
            # save model
            new_model.save('saveAttentionWeights/saveModel_fold' + str(index + 1) + '.h5')

            # save model
        new_model.save('saveAttentionWeights/saveModel_fold' + str(index + 1) + '.h5')

