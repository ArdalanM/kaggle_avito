# coding: utf8
__author__ = 'Ardalan'

import gensim, logging, os, argparse
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--model",default="xxx.model",help="A trained w2v model")
parser.add_argument("--qry",default="boobs",help="A query string to evaluate")
parser.add_argument("--top",type=int, default="10",help="Nb result words")
args = parser.parse_args()

# filename = '/home/ardalan/Documents/iosquare/moderation_match_meetic/data/0_raw_datasets/meetic.txt'
model_filename = args.model
qry = args.qry
top = args.top

if __name__ == '__main__':

    model = gensim.models.Word2Vec.load(model_filename)
    print(model.most_similar([qry], topn=top))
