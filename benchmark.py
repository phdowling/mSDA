__author__ = 'dowling'
logFormat = '%(asctime)s %(levelname)-8s %(name)-18s: %(message)s'
import logging
logging.basicConfig(format=logFormat, level=logging.DEBUG)
ln = logging.getLogger(__name__)

from reuters import stream_reuters_documents

from collections import defaultdict

from linear_msda import mSDA
from gensim.models import LsiModel
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full, full2sparse

from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier
import numpy as np

dictionary = None

settings = {
    "models": {"lsi": False, "msda": True, "noise": True, "bow": True},
    "dimensionalities": {
        "lsi": 200,
        "msda": 1000
    },
    "held_out_docs": 2000,
    #"classifier": "PassiveAggressive"
    "classifier": "Perceptron"

}

class ReutersCorpus(object):
    def get_documents(self):
        for document in stream_reuters_documents():
            yield simple_preprocess(document["content"])

    def __iter__(self):
        return self.get_documents()


class BOWmodel(object):
    def __getitem__(self, item):
        return item


class NoiseModel(object):
    def __init__(self, dims):
        self.dims = dims

    def __getitem__(self, item):
        return full2sparse(np.random.randn(1, self.dims))

class bow_corpus():
    def __iter__(self):
        for doc in ReutersCorpus():
            yield dictionary.doc2bow(doc)


def train_models():
    models = dict()
    if settings["models"]["msda"]:
        dims = settings["dimensionalities"]["msda"]
        try:
            msda = mSDA.load("reuters_msda_%sdims" % dims)
            # the line below is for testing a model I have locally on my machine
            #msda = mSDA.load("persist/mSDA/mSDA_wiki_dim-1000_stem-False_tfidf-False_noise-0.5_num_layers-3")
        except:
            ln.info("Training mSDA...")

            prototype_ids = [id_ for id_, freq in sorted(dictionary.dfs.items(), key=lambda (k, v): v, reverse=True)[:dims]]
            msda = mSDA(0.5, 5, len(dictionary), dims, prototype_ids=prototype_ids)
            msda.train(bow_corpus())
            msda.save("reuters_msda_%sdims" % dims)
        msda.__out_size = dims
        models["msda"] = msda

    if settings["models"]["lsi"]:
        dims = settings["dimensionalities"]["lsi"]
        try:
            lsi = LsiModel.load("reuters_lsi_%sdims" % dims)
        except:
            ln.info("Training LSI...")
            lsi = LsiModel(corpus=bow_corpus(), num_topics=dims, id2word=dictionary)
            lsi.save("reuters_lsi_%sdims" % dims)
        lsi.__out_size = dims
        models["lsi"] = lsi

    return models


def train_classifiers(models, train_data):
    classifiers = dict()
    for modelname, model in models.items():

        if settings["classifier"] == "Perceptron":
            classifier = Perceptron()
        if settings["classifier"] == "PassiveAggressive":
            classifier = PassiveAggressiveClassifier()

        for sample_no, (text, is_acq) in enumerate(train_data):
            bow = dictionary.doc2bow(simple_preprocess(text))

            model_features = sparse2full(model[bow], model.__out_size)
            label = np.array([is_acq])
            #ln.debug("%s, %s "% (model_features, label.shape))

            classifier.partial_fit(model_features, label, classes=np.array([True, False]))
            if sample_no % 500 == 0:
                ln.debug("Classifier for %s trained %s samples so far." % (modelname, sample_no))

        classifiers[modelname] = classifier
        ln.info("Finished training classifier for %s" % modelname)

    return classifiers


def run_evaluation(classifiers, models, eval_samples):
    ln.info("Beginning evaluation")
    classifications = dict()
    for modelname, classifier in classifiers.items():
        model = models[modelname]
        model_classifications = defaultdict(int)
        for sample_no, (eval_sample_text, actual_label) in enumerate(eval_samples):
            bow = dictionary.doc2bow(simple_preprocess(eval_sample_text))
            model_features = sparse2full(model[bow], model.__out_size)
            predicted_label = classifier.predict(model_features)[0]

            model_classifications[(actual_label, predicted_label)] += 1
            if sample_no % 500 == 0:
                ln.debug("Classifier for %s evaluated %s samples so far." % (modelname, sample_no))
        classifications[modelname] = model_classifications
    ln.info("Finished evaluation")
    return classifications


def output_results(classifications):
    ln.debug("settings: %s" % settings)
    ln.info("")
    ln.info("### EVALUATION RESULTS ###")
    for modelname in classifications:
        tp = classifications[modelname][(True, True)]
        fp = classifications[modelname][(False, True)]
        fn = classifications[modelname][(True, False)]
        tn = classifications[modelname][(False, False)]
        total = tp + fp + fn + tn
        accuracy = float((tp + tn)) / float(total)
        P = float(tp) / float(tp + fp)
        R = float(tp) / float(tp + fn)
        F1 = 2 * (P * R) / (P + R)

        ln.info("%s:" % modelname)
        ln.info("TP: %s\t FP: %s" % (tp, fp))
        ln.info("FN: %s\t TN: %s" % (fn, tn))
        ln.info("Total test samples: %s" % total)
        ln.info("Accuracy: %s" % accuracy)
        ln.info("P:%s \t R: %s" % (P, R))
        ln.info("F1: %s" % F1)
        ln.info("")
        ln.info("")


def main():
    global dictionary
    try:
        dictionary = Dictionary.load_from_text("persist/reuters_dictionary.txt")
        #dictionary = Dictionary.load_from_text("persist/wiki_stem-False_keep-100000_nobelow-20_noabove-0.1_wordids.txt.bz2")

    except:
        dictionary = Dictionary(ReutersCorpus())
        dictionary.filter_extremes()
        dictionary.save_as_text("persist/reuters_dictionary.txt")

    models = train_models()

    if settings["models"]["bow"]:
        bowmodel = BOWmodel()
        bowmodel.__out_size = len(dictionary)
        models["bow"] = bowmodel

    if settings["models"]["noise"]:
        noisemodel = NoiseModel(1000)
        noisemodel.__out_size = 1000
        models["noise"] = noisemodel

    num_train_samples = 21578 - settings["held_out_docs"]
    test_samples = []


    class generate_train_samples(object):
        first_iteration = True

        def __iter__(self):
            count = 0
            for document in stream_reuters_documents():
                sample = document["content"], "acq" in document["topics"]  # todo: maybe try "usa" or "earn"
                if count > num_train_samples:
                    if self.first_iteration:
                        test_samples.append(sample)
                else:
                    yield sample
                count += 1
            self.first_iteration = False

    classifiers = train_classifiers(models, generate_train_samples())

    classifications = run_evaluation(classifiers, models, test_samples)
    #output_results(classifications)

    return classifications

if __name__ == "__main__":
    main()

### MISCELLANEOUS STUFF, not important. ###

def get_topic_counts():
    labels = defaultdict(int)
    for document in stream_reuters_documents():
        for topic in document["topics"]:
            labels[topic] += 1
    return labels

