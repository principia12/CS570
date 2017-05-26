import json
import clstm
import shorttext
import sys
from keras.datasets import imdb
import keras.preprocessing.text

reload(sys)
sys.setdefaultencoding('utf8')



def parse_json(filename, numData):
    data = {}
    cnt = 0
    with open(filename) as f:
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

            if (numData != None) and (cnt == numData):
                return data
            
    return data

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

def get_word_index(path):
    """Retrieves the dictionary mapping word indices back to words.
    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).
    # Returns
        The word index dictionary.
    """
    f = open(path)
    data = json.load(f)
    f.close()
    return data

def getSentence(wordIdx, data):
    '''
    data shape: numpy ndarray
    return: list consist of sentences
    '''
    res = []
    for lst in data:
        temp = ''
        for item in lst:
            word = wordIdx.keys()[wordIdx.values().index(item)]
            temp = temp + " "+word
        # print temp
        res.append(temp)
        
    return res


def parseText(filename):
    '''
    input: imdb-dev.txt.ss
    output: dictionary. key: sentiment information (3rd column). value: review data (4th column)
    '''
    res = {}
    with open(filename) as f:
        for line in f.readlines():
            temp = line.strip().split('\t\t')
            rating = int(temp[2])
            review = temp[3].replace('<sssss>', ' ')
            if rating in res:
                res[rating].append(review)
            else:
                res[rating] = [review]

    for i in range(1,11):
        if len(res[i]) > 100:
            res[i] = res[i][:100]
            print 'len(res[%d]) is over 100' % i
    return res


if __name__ == "__main__":
    jsonName = '../data/yelp_academic_dataset_review.json'
    txtName = '../data/imdb-dev.txt.ss'
    path = '../data/imdb_word_index.json'
    # data = parse_json(jsonName, 12500)
    print 'parseText function starts'
    trainclassdict = parseText(txtName)
    
    # wordIdx = get_word_index(path)
    # print('getSentence function starts')
    # x_trainSentences = getSentence(wordIdx, x_train)
    
    # print(x_trainSentences)
    print 'wvmodel starts'
    wvmodel = shorttext.utils.load_word2vec_model('../data/GoogleNews-vectors-negative300.bin.gz')
    #trainclassdict = shorttext.data.subjectkeywords()
    # trainclassdict = parse_json('./yelp_academic_dataset_review.json')
    print 'wvmodel ends'
    print 'kmodel starts'
    kmodel = clstm.CLSTMWordEmbed(len(trainclassdict.keys()))
    print 'kmodel ends'
    print 'classifier starts'
    classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(wvmodel)
    print 'classifier training starts'
    classifier.train(trainclassdict, kmodel)
    print 'while loop starts'
    while True:
        text = raw_input('Input short text to classify: ')
        text = text.strip()
        if text == '-quit':
            break
        else:
            print classifier.score(text)
