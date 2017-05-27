from keras.layers import Convolution1D, MaxPooling1D, Dropout, GRU, Dense
from keras.models import Sequential
from keras.regularizers import l2
from keras import optimizers

# C-LSTM
# Chunting Zhou, Chonglin Sun, Zhiyuan Liu, Francis Lau,
# "A C-LSTM Neural Network for Text Classification", arXiv:1511.08630 (2015).
def CLSTMWordEmbed(nb_labels,
                   nb_filters=1200,
                   n_gram=2,
                   maxlen=15,
                   vecsize=300,
                   cnn_dropout=0.0,
                   nb_rnnoutdim=1200,
                   rnn_dropout=0.2,
                   final_activation='softmax',
                   dense_wl2reg=0.0,
                   optimizer='adam'):
    """ Returns the C-LSTM neural networks for word-embedded vectors.
    Reference: Chunting Zhou, Chonglin Sun, Zhiyuan Liu, Francis Lau,
    "A C-LSTM Neural Network for Text Classification,"
    (arXiv:1511.08630). [`arXiv
    <https://arxiv.org/abs/1511.08630>`_]
    :param nb_labels: number of class labels
    :param nb_filters: number of filters (Default: 1200)
    :param n_gram: n-gram, or window size of CNN/ConvNet (Default: 2)
    :param maxlen: maximum number of words in a sentence (Default: 15)
    :param vecsize: length of the embedded vectors in the model (Default: 300)
    :param cnn_dropout: dropout rate for CNN/ConvNet (Default: 0.0)
    :param nb_rnnoutdim: output dimension for the LSTM networks (Default: 1200)
    :param rnn_dropout: dropout rate for LSTM (Default: 0.2)
    :param final_activation: activation function. Options: softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear. (Default: 'softmax')
    :param dense_wl2reg: L2 regularization coefficient (Default: 0.0)
    :param optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: adam)
    :return: keras sequantial model for CNN/ConvNet for Word-Embeddings
    :type nb_labels: int
    :type nb_filters: int
    :type n_gram: int
    :type maxlen: int
    :type vecsize: int
    :type cnn_dropout: float
    :type nb_rnnoutdim: int
    :type rnn_dropout: float
    :type final_activation: str
    :type dense_wl2reg: float
    :type optimizer: str
    :rtype: keras.model.Sequential
    """
    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filters,
                            filter_length=n_gram,
                            border_mode='valid',
                            activation='relu',
                            input_shape=(maxlen, vecsize)))
    if cnn_dropout > 0.0:
        model.add(Dropout(cnn_dropout))
    model.add(MaxPooling1D(pool_length=maxlen - n_gram + 1))
    model.add(GRU(nb_rnnoutdim))
    if rnn_dropout > 0.0:
        model.add(Dropout(rnn_dropout))
    model.add(Dense(nb_labels,
                    activation=final_activation,
                    W_regularizer=l2(dense_wl2reg)
                    )
              )

    adam = optimizers.Adam(decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    return model