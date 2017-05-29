import nltk
import string
import numpy as np
import gensim
import random

from keras.layers import Convolution1D, MaxPooling1D, Dropout, LSTM, Dense, Flatten, GRU, Reshape
from keras.models import Sequential
from keras.regularizers import l2
from keras import optimizers


BATCH_SIZE = 500
w2vecmodel = None

def parseText(filename, nb_lables=5):

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

            curr_labels = [0.0]*nb_lables
            for i in range(nb_lables):
                if (rating > 1):
                    t = 111
                if i == (rating-1):
                    curr_labels[i] = 1.0
                    break


            sentences.append(list_of_words)
            labels.append(curr_labels)

    return (sentences, np.array(labels))


def parse_text_to_review(path):
    data = []
    cnt = 0
    with open(path, 'r', encoding='utf8') as f:
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
    classlabels = 5
    cnt = 0
    while 1:
        random.shuffle(classdict)
        for review in classdict:
            category_bucket = [0.0] * classlabels
            category_bucket[review[1]] = 1.0

            if cnt % BATCH_SIZE == 0:
                x_train = np.zeros(shape=(BATCH_SIZE, 150, 300))
                y_train = []

            x_train[cnt % BATCH_SIZE] = convert_to_w2vec_matrix([review[0]])[0]
            y_train.append(category_bucket)

            cnt += 1
            if cnt % BATCH_SIZE == 0:
                y_train = np.array(y_train, dtype=np.int)
                yield (x_train, y_train)


def convert_to_w2vec_matrix(sent_list,model=None, max_word_len=150):
    global w2vecmodel
    model = w2vecmodel
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
           nb_filters=300,
           n_gram=3,
           maxlen=150,
           vecsize=300,
           cnn_dropout=0.5,
           nb_rnnoutdim=300,
           rnn_dropout=0.5,
           final_activation='softmax',
           dense_wl2reg=0.005,
           optimizer='adam'):


    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filters,
                            filter_length=n_gram,
                            border_mode='valid',
                            activation='relu',
                            input_shape=(maxlen, vecsize)))
    if cnn_dropout > 0.0:
        model.add(Dropout(cnn_dropout))
    print ("after convolution output : ", model.output_shape)
    model.add(MaxPooling1D(pool_length=maxlen - n_gram + 1))
    print("after pooling output : ", model.output_shape)
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


def main():
    global w2vecmodel
    global BATCH_SIZE

    classdict = parse_text_to_review('../data/yelp-2013-train.txt.ss')
    #classdict = classdict[0:1000]
    total_train = len(classdict)
    testdict = parse_text_to_review('../data/yelp-2013-test.txt.ss')
    #testdict = testdict[0:1000]
    total_test = len(testdict)

    print ('Data load finished')

    w2vecmodel = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    print( 'word2vec load finished...')


    wl2reg_range = [0.001, 0.005, 0.01]
    dropout_val_range = [0.2, 0.5]
    nb_filters_range = [150, 300]

    score_results = []

    for dropout_val in dropout_val_range:
        for reg_val in wl2reg_range:
            for nb_filters_val in nb_filters_range:
                if (reg_val == 0.005 and dropout_val == 0.5 and nb_filters_val == 300):
                    continue
                if (reg_val == 0.01 and dropout_val == 0.2 and nb_filters_val == 300):
                    continue
                model = C_GRNN(nb_labels=5,
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
                    for i in range(5):
                        if max_prob < score[i]:
                            max_prob = score[i]
                            max_class = i

                    if review[1] == max_class:
                        cnt_correct += 1
                    cnt_all += 1

                    if(cnt_all % 2000 == 0):
                        print("%d test finished.."%(cnt_all))


                print (cnt_correct/cnt_all)
                score_results.append((dropout_val, reg_val, nb_filters_val, cnt_correct/cnt_all))

                print('######## one test finished', score_results[-1])

    for result in score_results:
        print(result)

    print('finished!')


if __name__ == "__main__":
    main()

    '''
    model.fit(x_train, y_train,
              batch_size=1024,
              epochs=10)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''
