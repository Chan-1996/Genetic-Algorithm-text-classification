

# import keras
from keras.datasets import mnist, cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten,Input, Embedding, Conv1D, GlobalMaxPooling1D, concatenate
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam, RMSprop, Adadelta, SGD

import logging
from process_Data import main
import numpy as np
# Helper: Early stopping.
# early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto')


# patience=5)
# monitor='val_loss',patience=2,verbose=0



def get_data(data):

    # get data
    x_train, y_train, x_test, y_test, max_len, vocab_size = main(data)
    if data == "MR":
        epochs = 5
        nb_classes = 2
        embedding_matrix = np.load('word2vec_mr.npy')
    else:
        epochs = 4
        nb_classes = 6
        embedding_matrix = np.load('word2vec_terc.npy')
    print(x_train.shape, y_train.shape)
    # x._train shape: (5452, 38, 300)
    # input shape (38, 300)
    input_shape = x_train.shape[1:]

    return (nb_classes, input_shape, x_train, x_test, y_train, y_test, epochs, embedding_matrix, vocab_size)


def compile_model_cnn(genome, nb_classes, input_shape, vocab_size, embedding_weights):

    # Get our network parameters.
    filter = genome.geneparam['filter']
    filter1 = genome.geneparam['filter1']
    filter2 = genome.geneparam['filter2']
    linear_dim = genome.geneparam['linear_dim']
    activation = genome.geneparam['activation']
    optimizer = genome.geneparam['optimizer']
    lr = genome.geneparam['lr']
    batch = genome.geneparam['batch_size']
    dropout = genome.geneparam['dropout']

    logging.info("Architecture:%s,%s,%s,%s,%s,%s" % (str([filter, filter1, filter2]), activation, optimizer, linear_dim,lr, batch))

    max_len = input_shape[0]
    inputs = Input(shape=input_shape)
    embedding = Embedding(vocab_size, 300, input_length=max_len, weights=[embedding_weights])(inputs)

    cnn1 = Conv1D(filter, 3, padding='same', strides=1, activation=activation)(embedding)
    maxpooling1 = GlobalMaxPooling1D()(cnn1)
    cnn2 = Conv1D(filter1, 4, padding='same', strides=1, activation=activation)(embedding)
    maxpooling2 = GlobalMaxPooling1D()(cnn2)
    cnn3 = Conv1D(filter2, 5, padding='same', strides=1, activation=activation)(embedding)
    maxpooling3 = GlobalMaxPooling1D()(cnn3)

    maxpooling = concatenate([maxpooling1, maxpooling2, maxpooling3], axis=-1)

    final = Dropout(dropout)(maxpooling)
    # final = Dense(linear_dim, activation=activation)(drop)
    outputs = Dense(nb_classes, activation='softmax')(final)
    model = Model(inputs=inputs, outputs=outputs)

    if optimizer == "adam":
        optimizer = Adam(lr=lr)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(lr=lr)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr=lr)
    else:
        optimizer = SGD(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train_and_score(genome, data='MR'):
    """Train the model, return test loss.


    """
    logging.info("Getting Keras datasets")


    nb_classes, input_shape, x_train, x_test, y_train, y_test, epochs, embedding_matrix, vocab_size = get_data(data)

    logging.info("Compling Keras model")

    model = compile_model_cnn(genome, nb_classes, input_shape, vocab_size, embedding_matrix)

    history = LossHistory()
    batch_size = genome.geneparam['batch_size']
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              # using early stopping so no real limit - don't want to waste time on horrible architectures
              verbose=2,
              validation_data=(x_test, y_test),
              # callbacks=[history])
              )

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    K.clear_session()


    return score[1]  # 1 is accuracy. 0 is loss.
