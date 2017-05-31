import os
import urllib
from zipfile import ZipFile

URL = 'https://archive.junheecho.com/cs570/'
DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(DIR, 'data')

GOOGLE_NEWS_VECTORS = 'GoogleNews-vectors-negative300.bin.gz'

EMNLP_2015 = 'emnlp-2015-data.zip'

IMDB_TRAIN = 'imdb-train.txt.ss'
IMDB_TEST = 'imdb-test.txt.ss'

YELP_TRAIN = 'yelp-2013-train.txt.ss'
YELP_TEST = 'yelp-2013-test.txt.ss'

YELP_ACADEMIC_DATASET = 'yelp-dataset-challenge-round9.zip'
YELP_ACADEMIC_DATASET_JSON = 'yelp_academic_dataset_review.json'

def file_path(filename):
    return os.path.join(DATA_DIR, filename)

def load():
    def download(filename, unzip=False):
        if not os.path.exists(file_path(filename)):
            urllib.urlretrieve(URL + filename, file_path(filename))
            if unzip:
                zipfile = ZipFile(file_path(filename), 'r')
                zipfile.extractall(DATA_DIR)
                zipfile.close()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Google News Vectors
    print("Downloading Google News Vectors...")
    download(GOOGLE_NEWS_VECTORS)

    # Yelp Academic Dataset
    print("Downloading IMDB and Yelp Academic Dataset...")
    download(EMNLP_2015, unzip=True)
