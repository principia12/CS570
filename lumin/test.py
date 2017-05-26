import json
import clstm
import shorttext
import sys

reload(sys)
sys.setdefaultencoding('utf8')

data = {}
cnt = 0
with open('../data/yelp_academic_dataset_review.json') as f:
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

        if cnt == 10000:
            break

        
print('wvmodel starts')
wvmodel = shorttext.utils.load_word2vec_model('../data/GoogleNews-vectors-negative300.bin.gz')
print('wvmodel closed')
print('clstm starts')
kmodel = clstm.CLSTMWordEmbed(len(data.keys()))
print('clstm closed')
print(kmodel)
print('classifier starts')
classifier = shorttext.classifiers.VarNNEmbeddedVecClassifier(wvmodel)
classifier.train(data, kmodel)
print('classifier closed')
print(classifier.score('fuck you'))

# print("very negative:", len(data['very negative']))
# print("negative:", len(data['negative']))
# print("neutral:", len(data['neutral']))
# print("positive:", len(data['positive']))
# print("very positive:", len(data['very positive']))
        
