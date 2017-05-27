import shorttext
import clstm
import json
import sys
from keras.datasets import imdb
from shorttext.utils import tokenize
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')

BATCH_SIZE = 50

def parse_json(path):
    data = {}
    with open(path) as f:
        cnt = 0
        for line in f:
            cnt += 1
            star = json.loads(line)['stars']
            review = json.loads(line)['text'].encode('utf-8')
            if star == 1:
                temp = 'very negative'
            elif star == 2:
                temp = 'negative'
            elif star == 3:
                temp = 'neutral'
            elif star == 4:
                temp = 'positive'
            else:
                temp = 'very positive'

            if temp in data:
                data[temp].append(review)
            else:
                data[temp] = [review]

    return data

def get_word(x):
    for word, index in word_index_dict.items():
        if index == x:
            return word

    return "_error"

def parse_imdb():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=400)
    x_test = sequence.pad_sequences(x_test, maxlen=400)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    word_index_dict = imdb.get_word_index()

    x_train = [' '.join([get_word(x) for x in train]) for train in x_train]
    x_test = [' '.join([get_word(x) for x in test]) for test in x_test]


class DataYieldVarNNEmbeddedVecClassifier(shorttext.classifiers.VarNNEmbeddedVecClassifier):
    def convert_trainingdata_matrix_yield(self, classdict):
        """ Convert the training data into format put into the neural networks.
        Convert the training data into format put into the neural networks.
        This is called by :func:`~train`.
        :param classdict: training data
        :return: a tuple of three, containing a list of class labels, matrix of embedded word vectors, and corresponding outputs
        :type classdict: dict
        :rtype: (list, numpy.ndarray, list)
        """
        classlabels = classdict.keys()
        lblidx_dict = dict(zip(classlabels, range(len(classlabels))))

        # tokenize the words, and determine the word length
        #phrases = []
        #indices = []

        cnt = 0
        while 1:
			for label in classlabels:
				for shorttext in classdict[label]:
					shorttext = shorttext if type(shorttext) == str else ''
					category_bucket = [0] * len(classlabels)
					category_bucket[lblidx_dict[label]] = 1

					#indices.append(category_bucket)
					#phrases.append(tokenize(shorttext))
					
					phrases_i = tokenize(shorttext)

					if cnt % BATCH_SIZE == 0:
						x_train = np.zeros(shape=(BATCH_SIZE, self.maxlen, self.vecsize))
						y_train = []

					for j in range(min(self.maxlen, len(phrases_i))):
						x_train[cnt % BATCH_SIZE, j] = self.word_to_embedvec(phrases_i[j])
					y_train.append(category_bucket)
					
					cnt += 1
					if cnt % BATCH_SIZE == 0:
						y_train = np.array(y_train, dtype=np.int)
						yield (x_train, y_train)

        # store embedded vectors
        #indices = np.array(indices, dtype=np.int)
        #train_embedvec = np.zeros(shape=(len(phrases), self.maxlen, self.vecsize))
        #while 1:
        #    for i in range(len(phrases)):
        #        for j in range(min(self.maxlen, len(phrases[i]))):
        #            train_embedvec[i, j] = self.word_to_embedvec(phrases[i][j])
        #        yield (train_embedvec[i:i+1], indices[i:i+1])

    def train(self, classdict, kerasmodel, nb_epoch=10):
        """ Train the classifier.
                The training data and the corresponding keras model have to be given.
                If this has not been run, or a model was not loaded by :func:`~loadmodel`,
                a `ModelNotTrainedException` will be raised.
                :param classdict: training data
                :param kerasmodel: keras sequential model
                :param nb_epoch: number of steps / epochs in training
                :return: None
                :type classdict: dict
                :type kerasmodel: keras.models.Sequential
                :type nb_epoch: int
                """
        # convert classdict to training input vectors
        self.classlabels = classdict.keys()
        total_train = sum([len(classdict[cls]) for cls in classdict])
        #train_embedvec, indices = self.convert_trainingdata_matrix(classdict)

        # train the model
        #kerasmodel.fit(train_embedvec, indices, epochs=nb_epoch)
        kerasmodel.fit_generator(self.convert_trainingdata_matrix_yield(classdict),
                                 steps_per_epoch=int(total_train/BATCH_SIZE), epochs=nb_epoch)

        # flag switch
        self.model = kerasmodel
        self.trained = True

if __name__ == '__main__':
    wvmodel = shorttext.utils.load_word2vec_model('GoogleNews-vectors-negative300.bin.gz')
    #trainclassdict = shorttext.data.subjectkeywords()
    trainclassdict = parse_json('./yelp_academic_dataset_review.json')
    kmodel = clstm.CLSTMWordEmbed(len(trainclassdict.keys()), n_gram=3, maxlen=150, rnn_dropout=0.5, dense_wl2reg=0.001)
    #classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(wvmodel)
    classifier = DataYieldVarNNEmbeddedVecClassifier(wvmodel, maxlen=150)
    classifier.train(trainclassdict, kmodel)
    while True:
        text = raw_input('Input short text to classify: ')
        text = text.strip()
        if text == '-quit':
            break
        else:
            print classifier.score(text)