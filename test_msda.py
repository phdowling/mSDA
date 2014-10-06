__author__ = 'dowling'

from bs4 import BeautifulSoup
import logging
ln = logging.getLogger("mSDA")
ln.setLevel(logging.DEBUG)

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(name)-18s: %(message)s')

from stopwords import stopwords

from stemming.porter2 import stem
import string

import gensim

from linear_msda import mSDA, mSDAhd

import os

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

class Reuters21578DataSource():
    def __init__(self):
        self.reutersFiles = []
        self.loadFiles = ["reuters21578/reut2-00%s.sgm" % str(num) for num in range(10)] + \
                         ["reuters21578/reut2-0%s.sgm" % str(num) for num in range(10, 22)]
        self.updateCount = 0
        self.updating = False

    def getDocuments(self):
        return []

    def updateAndGetDocuments(self):  # pretend we're downloading the rest of the corpus
        self.updating = True
        try:
            filename = self.loadFiles[self.updateCount]
        except:
            return []
        count = 0
        documents = []
        with open(filename, "r") as f:
            data = f.read()
            soup = BeautifulSoup(data)
            contents = soup.find_all("text")
            for cont in contents:
                count += 1
                d = Document(cont.text)
                d.sourceType = self.__class__.__name__
                documents.append(d)
        ln.info("on updating, got %s documents from file %s." % (count, filename))
        self.updateCount += 1
        self.updating = False
        return documents


class Document(object):
    def __init__(self, text):
        self.text = text
        self.id = None
        self.preprocessed = []
        self.vectors = dict()

    def __iter__(self):
        for id, token in self.preprocessed:
            yield token

    def __len__(self):
        return len(self.preprocessed)



# SETTINGS
STOPWORDS = set(stopwords)


class TokenizingPorter2Stemmer():
    def __init__(self, stopWords=STOPWORDS):
        self.stopWords = set(map(stem, stopWords))
        #self.stopWords = set(self.stemWords(self.stopWords))

    def preprocess(self, doc, dictionary, allow_update=True):
        if isinstance(doc, str):
            text = doc
        else:
            text = doc.text[:]
        text = text.encode("ascii", "ignore").lower()
        text = text.split()
        text = self.removeHTML(text)
        text = self.removePunctuation(text)
        #text = self.stemWords(text)
        text = self.removeNonsense(text)
        text = self.removeStopWords(set(text))
        text = sorted(text, key=lambda x: -len(x))
        text = dictionary.doc2bow(text, allow_update=allow_update)

        return text
        #else:
        #    doc.preprocessed = text

    def removeHTML(self, text):
        for term in text[:]:
            if term.startswith("<") and term.endswith(">"):
                text.remove(term)
        return text

    def removeNonsense(self, text):
        def isNonsense(term):
            if term[:4] == "http" or term[:4] == "href" or term[:7] == "srchttp":
                return True
            if any((x in term for x in "1234567890")):
                return True
            if len(term) > 50:
                    return True
            return False
        return [term for term in text if not isNonsense(term)]

    def removePunctuation(self, text):
        return [term.translate(string.maketrans("",""), string.punctuation) for term in text]

    def removeStopWords(self, textSet):
        return textSet - self.stopWords

    def stemWords(self, text):
        return map(stem, text)

ln.debug("load corpus and preprocess")
data = Reuters21578DataSource()
preprocessor = TokenizingPorter2Stemmer()
dictionary = gensim.corpora.dictionary.Dictionary()
all_preprocessed = []
for _ in range(30):
    docs = [preprocessor.preprocess(doc, dictionary) for doc in data.updateAndGetDocuments()]
    all_preprocessed += docs

ln.debug("got %s documents " % len(all_preprocessed))
ln.debug("dictionary has %s unique terms" % len(dictionary))

ln.debug("find most frequent terms")
most_frequent_ids = dictionary.dfs.items()[:]
most_frequent_ids.sort(key=lambda (key, val): -val)

inv_map = {v: k for k, v in dictionary.token2id.items()}
k = 3000
top_k = [k for k, v in most_frequent_ids[:k]]

#print [inv_map[k] for k in top_k]

ln.debug("train mSDA")
msda = mSDAhd(top_k, len(dictionary), noise=0.5, num_layers=5)
#msda = mSDA(noise=0.5, num_layers=3, input_dimensionality=len(dictionary))

representations = msda.train(all_preprocessed, return_hidden=True)
msda.save("reuters21578_3000dim_nostem")


def get_reps(string1):
    bow1 = preprocessor.preprocess(string1, dictionary, allow_update=False)
    reps1 = msda.get_hidden_representations([bow1])
    return reps1

