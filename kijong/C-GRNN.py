import nltk
import gensim
import string
import numpy as np

from keras.layers import Convolution1D, MaxPooling1D, Dropout, LSTM, Dense, Flatten
from keras.models import Sequential
from keras.regularizers import l2
from keras import optimizers


def parseText(filename):

    sentences = []
    labels = []
    count = 0
    with open(filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            count += 1
            temp = line.strip().split('\t\t')
            rating = int(temp[2])

            review = temp[3].replace('<sssss>', '').lower()

            tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            list_of_words = tokenizer.tokenize(review)

            sentences.append(list_of_words)
            labels.append(rating-1)

            if ((count % 10000) is 0):
                print("%d data is loaded"%(count))
            if (count >= 5000):
                break

    return (sentences, labels)


def convert_to_w2vec_matrix(sent_list,model, max_word_len=150):
    train_x = np.zeros((len(sent_list),max_word_len,300))
    i = 0
    for sents in sent_list:
        j = 0
        for word in sents:
            if (word in model.vocab):
                train_x[i][j] = model[word]
            j += 1
            if (j >= max_word_len):
                break
    return train_x


def C_GRNN(nb_labels,
           x_train,
           y_train,
           nb_filters=300,
           n_gram=3,
           maxlen=150,
           vecsize=300,
           cnn_dropout=0.0,
           nb_rnnoutdim=300,
           rnn_dropout=0.2,
           final_activation='softmax',
           dense_wl2reg=0.0,
           optimizer='adam'):


    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filters,
                            filter_length=n_gram,
                            border_mode='valid',
                            activation='relu',
                            input_shape=(maxlen, vecsize)))
    if cnn_dropout > 0.0:
        model.add(Dropout(cnn_dropout))
    model.add(MaxPooling1D(pool_length=maxlen - n_gram + 1))
    model.add(Flatten())
    model.add(LSTM(nb_rnnoutdim))
    if rnn_dropout > 0.0:
        model.add(Dropout(rnn_dropout))
    model.add(Dense(nb_labels,
                    activation=final_activation,
                    W_regularizer=l2(dense_wl2reg)
                    )
              )

    adam = optimizers.Adam(decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    model.fit(x_train, y_train,
              batch_size=1024,
              epochs=10)


def main():

    train_sents, train_y = parseText('../data/yelp-2013-train.txt.ss')

    w2VecModel = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    print( 'word2vec load finished...')

    train_x = convert_to_w2vec_matrix(train_sents, w2VecModel)
    print('matrix conversion finished...')

    C_GRNN(5, train_x, train_y)
    print('finished')


if __name__ == "__main__":
    main()

