import shorttext
import clstm

if __name__ == '__main__':
    wvmodel = shorttext.utils.load_word2vec_model('GoogleNews-vectors-negative300.bin.gz')
    trainclassdict = shorttext.data.subjectkeywords()
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