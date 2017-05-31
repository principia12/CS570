import os
import argparse

from utils import data
from C_LSTM.run_yelp2013 import main as C_LSTM_MAIN
from C_GRNN.C_GRNN import main as C_GRNN_MAIN

from utils.data import file_path, GOOGLE_NEWS_VECTORS, IMDB_TRAIN, IMDB_TEST, YELP_TRAIN, YELP_TEST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A sentiment classification.')
    parser.add_argument('model', choices=['c-lstm', 'c-grnn'], help='a model, C-LSTM or C-GRNN.')
    parser.add_argument('data', choices=['imdb', 'yelp'], help='a dataset, IMDB or Yelp 2013.')
    parser.add_argument('-b', '--batch-size', type=int, default=100, help='a batch size.')

    args = parser.parse_args()
    print "Using " + args.model.upper() + " and",

    train, test, max_rating = None, None, 0
    batch_size = args.batch_size
    if (args.data == 'imdb'):
        print "IMDB data."
        train, test, max_rating = IMDB_TRAIN, IMDB_TEST, 10
    elif (args.data == 'yelp'):
        print "Yelp 2013 data."
        train, test, max_rating = YELP_TRAIN, YELP_TEST, 5

    DIR = os.path.dirname(os.path.realpath(__file__))
    data.load()

    print("Running " + args.model.upper() + "...")
    if args.model == 'c-lstm':
        C_LSTM_MAIN(train, test, max_rating, batch_size)
    elif args.model == 'c-grnn':
        C_GRNN_MAIN(train, test, max_rating, batch_size)
