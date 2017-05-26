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

    for i in range(1,6):
        if len(res[i]) > 10000:
            res[i] = res[i][:10000]
            print 'len(res[%d]) is over 100' % i
    return res

if __name__ == '__main__':
    wvmodel = shorttext.utils.load_word2vec_model('../data/GoogleNews-vectors-negative300.bin.gz')
    #trainclassdict = shorttext.data.subjectkeywords()
    #trainclassdict = parse_json('./yelp_academic_dataset_review.json')
    trainclassdict = parseText('../data/yelp/yelp-2013-train.txt.ss')
    kmodel = clstm.CLSTMWordEmbed(len(trainclassdict.keys()), n_gram=3, maxlen=150, rnn_dropout=0.5, dense_wl2reg=0.001)
    classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(wvmodel, maxlen=150)
    classifier.train(trainclassdict, kmodel, nb_epoch=20)

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