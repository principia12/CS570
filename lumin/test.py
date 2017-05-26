import json

data = {}
with open('../data/yelp_academic_dataset_review.json') as f:
    for line in f:
        star = json.loads(line)['stars']
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
            data[temp].append(json.loads(line)['text'])
        else:
            data[temp] = [json.loads(line)['text']]
        
