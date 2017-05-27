import shorttext
import clstm
import json
import sys
from shorttext.utils import tokenize
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')

BATCH_SIZE = 50

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
        print total_train
        #train_embedvec, indices = self.convert_trainingdata_matrix(classdict)

        # train the model
        #kerasmodel.fit(train_embedvec, indices, epochs=nb_epoch)
        kerasmodel.fit_generator(self.convert_trainingdata_matrix_yield(classdict),
                                 steps_per_epoch=int(total_train/BATCH_SIZE), epochs=nb_epoch)

        # flag switch
        self.model = kerasmodel
        self.trained = True

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
            if cnt == 100:
                break
    return data

def parseText(filename):
    '''
    input: imdb-dev.txt.ss
    output: dictionary. key: sentiment information (3rd column). value: review data (4th column)
    '''
    res = {}
    for i in range(1, 6):
        res[i] = []

    with open(filename) as f:
        for line in f.readlines():
            temp = line.strip().split('\t\t')
            rating = int(temp[2])
            review = temp[3].replace('<sssss>', ' ')

            #TODO : remove
            #if (rating > 1) and (rating < 5):
            #    continue

            if rating in res:
                res[rating].append(review)
            else:
                res[rating] = [review]

    #for i in range(1,6):
    #    if len(res[i]) > 10000:
    #        res[i] = res[i][:10000]
    #        print 'len(res[%d]) is over 100' % i
    return res

if __name__ == '__main__':
    wvmodel = shorttext.utils.load_word2vec_model('../data/GoogleNews-vectors-negative300.bin.gz')
    #trainclassdict = shorttext.data.subjectkeywords()
    #trainclassdict = parse_json('./yelp_academic_dataset_review.json')
    trainclassdict = parseText('../data/yelp/yelp-2013-train.txt.ss')
    kmodel = clstm.CLSTMWordEmbed(len(trainclassdict.keys()), n_gram=3, maxlen=150, rnn_dropout=0.5, dense_wl2reg=0.001)
    #classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(wvmodel, maxlen=150)
    classifier = DataYieldVarNNEmbeddedVecClassifier(wvmodel, maxlen=150)
    classifier.train(trainclassdict, kmodel, nb_epoch=10)

    testclassdict = parseText('../data/yelp/yelp-2013-test.txt.ss')
    cnt_correct = 0
    cnt_all = 0
    for key in testclassdict:
        for text in testclassdict[key]:
            score = classifier.score(text)
            max_prob = 0.0
            max_class = None
            for key_score in score:
                if max_prob < score[key_score]:
                    max_prob = score[key_score]
                    max_class = key_score
            if key == max_class:
                cnt_correct += 1
            cnt_all += 1

    print float(cnt_correct)/float(cnt_all)