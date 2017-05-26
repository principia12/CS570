import shorttext
import clstm
import json
import sys

reload(sys)
sys.setdefaultencoding('utf8')

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

class DataYieldVarNNEmbeddedVecClassifier(shorttext.classifiers.VarNNEmbeddedVecClassifier):
    def generate_trainingdata_from_json(self, path):
        with open(path) as f:
            for line in f:
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
        self.classlabels, train_embedvec, indices = self.convert_trainingdata_matrix(classdict)

        # train the model
        kerasmodel.fit(train_embedvec, indices, epochs=nb_epoch)
        #kerasmodel.fit_generator(self.generate_trainingdata_from_json('C:/MCDD/dataset/yelp_academic_dataset_review.json'),
        #                         steps_per_epoch=1000, epochs=nb_epoch)

        # flag switch
        self.model = kerasmodel
        self.trained = True

if __name__ == '__main__':
    wvmodel = shorttext.utils.load_word2vec_model('GoogleNews-vectors-negative300.bin.gz')
    #trainclassdict = shorttext.data.subjectkeywords()
    trainclassdict = parse_json('./yelp_academic_dataset_review.json')
    kmodel = clstm.CLSTMWordEmbed(len(trainclassdict.keys()))
    classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(wvmodel)
    classifier.train(trainclassdict, kmodel)
    while True:
        text = raw_input('Input short text to classify: ')
        text = text.strip()
        if text == '-quit':
            break
        else:
            print classifier.score(text)