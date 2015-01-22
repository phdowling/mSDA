__author__ = 'dowling'
# This test class has a lot of dependencies, mainly because I haven't had time to rewrite it in a cleaner way yet.
# However, using mSDA only requires numpy and scipy as a dependency.

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(name)-18s: %(message)s', level=logging.DEBUG)
ln = logging.getLogger("test_model")

from nltk.corpus import brown
from pattern.en import wordnet
from scipy.spatial.distance import cosine

from linear_msda import mSDAhd, mSDA

import random

from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim import matutils

from stemming.porter2 import stem
import string


class Preprocessor():
    def __init__(self, dictionary, use_stemming=False):
        self.use_stemming = use_stemming
        self.dictionary = dictionary

    def preprocess(self, text, allow_update=True, return_bow=True):
        text = text.encode("ascii", "ignore").lower()
        text = text.strip().split()
        text = [word.strip() for word in text]

        text = [term.translate(string.maketrans("", ""), string.punctuation) for term in text]
        if self.use_stemming:
            text = map(stem, text)

        if return_bow:
            text = self.dictionary.doc2bow(text, allow_update=allow_update)

        return text

def cosine_similarity(x, y):
    c = 1-cosine(x, y)
    if c < 0.0:
        return 0.0
    if c > 1.0:
        return 1.0
    else:
        return c

class mSDAWrapper(object):
    def __init__(self, filename, preprocessor):
        self.preprocessor = preprocessor
        self.model = mSDAhd.load(filename)
        self.output_dimensionality = self.model.output_dimensionality * ((self.model.num_layers + 1)
                                                                         if self.model.concatenate_output else 1)

    def compute_similarity(self, phrase1, phrase2):
        return cosine_similarity(matutils.sparse2full(self[phrase1], self.output_dimensionality),
                                 matutils.sparse2full(self[phrase2], self.output_dimensionality))

    def __getitem__(self, item):
        prep = self.preprocessor.preprocess(item, allow_update=False)
        return self.model[prep]

    @classmethod
    def train(cls, documents, dimensions, id2word, params):
        msdawrap = object.__new__(mSDAWrapper)
        msdawrap.model = mSDAhd(dimensions, id2word, noise=params["noise"], num_layers=params["num_layers"])
        msdawrap.model.train(documents, chunksize=10000)
        return msdawrap

    def save(self, fname):
        self.model.save(fname)


ln.info("preprocessing corpus")
dictionary = Dictionary()

preprocessor = Preprocessor(use_stemming=False, dictionary=dictionary)

corpusname = "brown"
corpus = [preprocessor.preprocess(" ".join(text), return_bow=True) for text in brown.sents()]
preprocessor.dictionary.filter_extremes(15, 0.1, 30000)
corpus = [preprocessor.preprocess(" ".join(text), allow_update=False, return_bow=True) for text in brown.sents()]

ln.debug("saving/loading corpus")
save = MmCorpus.serialize("test.mm", corpus)
corpus = MmCorpus("test.mm")


dimensions = 2000
params = [{"num_layers": 5, "noise": 0.7},
          {"num_layers": 3, "noise": 0.3}][0]

ln.info("training mSDA with %s dimensions. params: %s" % (dimensions, params))
model = mSDAWrapper.train(corpus, dimensions, dictionary, params)

paramstring = "_".join(["%s-%s" % (k, v) for k, v in params.items()])
savestring = "mSDA_%s_%s_" % (corpusname, paramstring)
model.save(savestring)
msda_wrapper = mSDAWrapper(savestring, preprocessor)

def get_synonyms(word):
    return [synset.synonyms for synset in wordnet.synsets(word)]

# run sanity checks
def generate_synonyms():
    synonyms = set()

    for termid in dictionary:
        term = dictionary[termid]
        if not term:
            continue
        synsets = get_synonyms(term)
        synset = [item for sublist in synsets for item in sublist]  # flatten
        for other_term in synset:
            other_term = " ".join(preprocessor.preprocess(other_term, allow_update=False, return_bow=False))
            if other_term != term:
                if other_term in dictionary.token2id:
                    synonyms.add((term, other_term))
    noise = zip([dictionary[random.randrange(len(dictionary))] for _ in range(len(synonyms))],
                [dictionary[random.randrange(len(dictionary))] for _ in range(len(synonyms))])
    return synonyms, noise


syns, noise = generate_synonyms()
sum_ = 0
for (term, syn) in syns:
    sum_ += msda_wrapper.compute_similarity(term, syn)

print "average synonym pair similarity: ", sum_ / len(syns)

sum_ = 0
for (term, ot) in noise:
    sum_ += msda_wrapper.compute_similarity(term, ot)

print "average noise pair similarity: ", sum_ / len(noise)