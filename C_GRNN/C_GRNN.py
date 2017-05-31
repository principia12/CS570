import sys
import nltk
import string
import numpy as np
import gensim
import random

from keras.layers import Convolution1D, MaxPooling1D, Dropout, LSTM, Dense,  GRU, GlobalAveragePooling1D
from keras.models import Sequential
from keras.regularizers import l2
from keras import optimizers

from utils.data import file_path, GOOGLE_NEWS_VECTORS, IMDB_TRAIN, IMDB_TEST, YELP_TRAIN, YELP_TEST

reload(sys)
sys.setdefaultencoding('utf8')

BATCH_SIZE = 200
max_len = 350
NB_LABLES = 10
w2vecmodel = None

def parse_text_to_review(path):
    data = []
    cnt = 0
    with open(path, 'r') as f:
        for line in f:
            temp = line.strip().split('\t\t')
            star = int(temp[2])
            review = temp[3].replace('<sssss>', ' ').lower()

            tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            list_of_words = tokenizer.tokenize(review)
            data.append((list_of_words,star-1))

            cnt += 1

    return data

def convert_trainingdata_matrix_yield(classdict):
    global BATCH_SIZE
    global max_len
    global NB_LABLES
    classlabels = NB_LABLES
    cnt = 0
    while 1:
        random.shuffle(classdict)
        for review in classdict:
            category_bucket = [0.0] * classlabels
            category_bucket[review[1]] = 1.0

            if cnt % BATCH_SIZE == 0:
                x_train = np.zeros(shape=(BATCH_SIZE, max_len, 300))
                y_train = []

            x_train[cnt % BATCH_SIZE] = convert_to_w2vec_matrix([review[0]])[0]
            y_train.append(category_bucket)

            cnt += 1
            if cnt % BATCH_SIZE == 0:
                y_train = np.array(y_train, dtype=np.int)
                yield (x_train, y_train)


def convert_to_w2vec_matrix(sent_list,model=None):
    global max_len
    model = w2vecmodel
    train_x = np.zeros((len(sent_list),max_len,300))
    i = 0
    for sents in sent_list:
        j = 0
        for word in sents:
            if (word in model.vocab):
                train_x[i][j] = model[word]
            j += 1
            if (j >= max_len):
                break
    return train_x


def C_GRNN(nb_labels,
           nb_filters=300,
           n_gram=3,
           vecsize=300,
           cnn_dropout=0.5,
           nb_rnnoutdim=300,
           rnn_dropout=0.5,
           final_activation='softmax',
           dense_wl2reg=0.005,
           optimizer='adam'):

    global max_len

    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filters,
                            filter_length=n_gram,
                            border_mode='valid',
                            activation='relu',
                            input_shape=(max_len, vecsize)))
    if cnn_dropout > 0.0:
        model.add(Dropout(cnn_dropout))
    print ("after convolution output : ", model.output_shape)
    #model.add(MaxPooling1D(pool_length=max_len - n_gram + 1))
    #print("after pooling output : ", model.output_shape)
    model.add(GRU(nb_rnnoutdim, return_sequences=True))
    print("after gru output : ", model.output_shape)
    model.add(GlobalAveragePooling1D())
    print("after pooling : ", model.output_shape)

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


def main(TRAIN, TEST, max_rating, batch_size):
    global w2vecmodel
    global BATCH_SIZE
    global NB_LABLES

    BATCH_SIZE = batch_size
    NB_LABLES = max_rating

    #classdict = parse_text_to_review('../data/yelp-2013-train.txt.ss')
    classdict = parse_text_to_review(file_path(TRAIN))
    total_train = len(classdict)

    #testdict = parse_text_to_review('../data/yelp-2013-test.txt.ss')
    testdict = parse_text_to_review(file_path(TEST))
    total_test = len(testdict)


    print ('Data load finished')

    w2vecmodel = gensim.models.KeyedVectors.load_word2vec_format(file_path(GOOGLE_NEWS_VECTORS), binary=True)
    print( 'word2vec load finished...')


    wl2reg_range = [0.001]
    dropout_val_range = [0.5]
    nb_filters_range = [150]

    score_results = []

    for dropout_val in dropout_val_range:
        for reg_val in wl2reg_range:
            for nb_filters_val in nb_filters_range:
                model = C_GRNN(nb_labels=NB_LABLES,
                               n_gram=3,
                               nb_filters=nb_filters_val,
                               nb_rnnoutdim=nb_filters_val,
                               cnn_dropout=dropout_val,
                               rnn_dropout=dropout_val,
                               dense_wl2reg=reg_val
                               )
                model.fit_generator(convert_trainingdata_matrix_yield(classdict),
                                    steps_per_epoch=int(total_train / BATCH_SIZE), epochs=15)

                print('training_finished')

                cnt_correct = 0
                cnt_all = 0

                for review in testdict:
                    x = convert_to_w2vec_matrix([review[0]])
                    score = model.predict(x)[0]
                    max_prob = 0.0
                    max_class = None
                    for i in range(NB_LABLES):
                        if max_prob < score[i]:
                            max_prob = score[i]
                            max_class = i

                    if review[1] == max_class:
                        cnt_correct += 1
                    cnt_all += 1

                    if(cnt_all % 1000 == 0):
                        print("%d test finished.."%(cnt_all))


                print (cnt_correct/cnt_all)
                score_results.append((dropout_val, reg_val, nb_filters_val, cnt_correct/cnt_all))

                print('######## one test finished', score_results[-1])

    for result in score_results:
        print(result)

    print('finished!')


if __name__ == "__main__":
    main(YELP_TRAIN, YELP_TEST, 5, 100)

    '''
    model.fit(x_train, y_train,
              batch_size=1024,
              epochs=10)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''
