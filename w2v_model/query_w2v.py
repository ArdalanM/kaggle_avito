uthor__ = 'Ardalan'

import gensim, logging, os, argparse
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--model",default="/media/shared_ardalan_evgeny/w2v_t_desc/w2v_size100_win5_mc5_sample0.001_iter5.model",help="A trained w2v model")
parser.add_argument("--qry",default="iphone",help="A query string to evaluate")
parser.add_argument("--top",type=int, default="20",help="Nb result words")
args = parser.parse_args()

filename = '/media/shared_ardalan_evgeny/w2v_t_desc/w2v_size100_win5_mc5_sample0.001_iter5.model'
model_filename = args.model
qry = args.qry
top = args.top

if __name__ == '__main__':
    model = gensim.models.Word2Vec.load(model_filename)
    print(model.most_similar([qry], topn=top))
