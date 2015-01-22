__author__ = 'dowling'
import logging
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

ln = logging.getLogger("mSDA")
ln.setLevel(logging.DEBUG)

from mda_layer import mDALayer

from gensim import utils, matutils
from gensim.corpora.mmcorpus import MmCorpus

import os


def convert(sparse_bow, dimensionality):
    dense = np.zeros((dimensionality, 1))
    for dim, value in sparse_bow:
        dense[dim] = value
    return csc_matrix(dense)


def convert_to_sparse_matrix(input_data, dimensionality):
    sparse = lil_matrix((dimensionality, len(input_data)))
    for docidx, document in enumerate(input_data):
        for word_id, count in document:
            sparse[word_id, docidx] = count
    return sparse.tocsc()


class mSDA(object):
    """
    Marginalized Stacked Denoising Autoencoder class.
    Probably don't want to initialize this directly, the provided utility classes are easier to deal with.
    """
    def __init__(self, noise, num_layers, input_dimensionality, output_dimensionality=None, prototype_ids=None):
        assert num_layers >= 1, "need at least one layer."

        self.lambda_ = 1e-05
        self.noise = noise
        self.input_dimensionality = input_dimensionality

        if output_dimensionality is None:
            self.output_dimensionality = input_dimensionality
        else:
            self.output_dimensionality = output_dimensionality

        reduction_layer = mDALayer(self.noise, self.lambda_, self.input_dimensionality, self.output_dimensionality,
                                   prototype_ids)

        self.reduction_layer = reduction_layer
        self.mda_layers = [mDALayer(noise, self.lambda_, output_dimensionality) for _ in range(num_layers - 1)]

    def train(self, corpus, chunksize=10000, use_temp_files=True):
        """
        train the underlying linear mappings.

        @param corpus is a gensim corpus compatible format
        @param use_temp_files determines whether to use temporary files to store the intermediate representations of
        the corpus to train the next layer. Setting flag True will not greatly affect memory usage, but will temporarily
        require a significant amount of disk space. Using temp files will strongly speed up training, especially as the
        number of layers increases.
        """
        ln.info("Training mSDA with %s layers. ")
        if not use_temp_files:
            ln.warn("Training without temporary files. May take a long time!")
            self.reduction_layer.train(corpus, chunksize=chunksize)
            current_representation = self.reduction_layer[corpus]
            for layer_num, layer in enumerate(self.mda_layers):

                # We feed the corpus through all intermediate layers to get the current representation
                # that representation is then used to train the next layer
                # this is memory-independent, but will probably be very slow.

                ln.info("Training layer %s.", layer_num)
                layer.train(current_representation, chunksize=chunksize)
                if layer_num < len(self.mda_layers) - 1:
                    current_representation = layer[current_representation]

        else:
            ln.info("Using temporary files to speed up training.")

            ln.info("Beginning training on %s layers." % (len(self.mda_layers) + 1))
            self.reduction_layer.train(corpus, chunksize=chunksize)

            # serializing intermediate representation
            MmCorpus.serialize(".msda_intermediate.mm", self.reduction_layer[corpus], progress_cnt=chunksize)

            # load corpus to train next layer
            current_representation = MmCorpus(".msda_intermediate.mm")

            for layer_num, layer in enumerate(self.mda_layers):
                layer.train(current_representation, chunksize=chunksize)
                os.remove(".msda_intermediate.mm")
                os.remove(".msda_intermediate.mm.index")

                if layer_num < len(self.mda_layers) - 1:
                    MmCorpus.serialize(".msda_intermediate.mm", layer[current_representation], progress_cnt=chunksize)
                    current_representation = MmCorpus(".msda_intermediate.mm")

        ln.info("mSDA finished training.")

    def _get_hidden_representations(self, input_data):
        """
        convert a numpy matrix of documents to their mSDA representation.
        if return_sparse is true, return the documents in list form, otherwise return a dense matrix
        if concatenate is true, the representation of each document is the concatenation of each layer output
            otherwise, use the last layers' output only
        """
        hidden = self.reduction_layer.__getitem__(input_data, numpy_input=True, numpy_output=True)
        for layer in self.mda_layers:
            hidden = layer.__getitem__(hidden, numpy_input=True, numpy_output=True)
        return hidden

    def __getitem__(self, bow, chunksize=10000):
        is_corpus, bow = utils.is_corpus(bow)
        if not is_corpus:
            bow = [bow]

        if chunksize:
            def transformed_corpus():
                for chunk_no, doc_chunk in utils.grouper(bow, chunksize):
                    chunk = matutils.corpus2dense(doc_chunk, self.input_dimensionality)
                    hidden = self._get_hidden_representations(chunk)
                    for column in hidden.T:
                        yield matutils.any2sparse(column)

        else:
            def transformed_corpus():
                for doc in bow:
                    yield matutils.any2sparse(
                        self._get_hidden_representations(matutils.corpus2dense(doc, self.input_dimensionality)))

        if not is_corpus:
            return list(transformed_corpus()).pop()
        else:
            return transformed_corpus()

    def save(self, filename_prefix):
        # need to save:
        #
        # type of msda
        # number of layers
        # noise level
        #
        # if msdahd:
        #   save randomized indices
        #   save each block's W matrix in the HD layer
        #
        # save each W matrix of the layers

        with open(filename_prefix, "w") as f:

            f.write("input_dimensionality=%s\n" % self.input_dimensionality)
            f.write("output_dimensionality=%s\n" % self.output_dimensionality)
            f.write("num_layers=%s\n" % (len(self.mda_layers) + 1,))

            f.write("noise=%s\n" % self.noise)

        np.save(filename_prefix + "_randidx", self.reduction_layer.randomized_indices)

        for idx, block in enumerate(self.reduction_layer.blocks):
            np.save(filename_prefix + "_block%s" % idx, block)
        for idx, layer in enumerate(self.mda_layers):
            assert len(layer.blocks) == 1, "Layer %s has %s layers (should be 1)." % (idx, len(layer.blocks))
            np.save(filename_prefix + "_layer%s" % idx, layer.blocks[0])

    @classmethod
    def load(cls, filename_prefix):
        # load metadata
        input_dimensionality = None
        output_dimensionality = None
        num_layers = None
        noise = None

        with open(filename_prefix, "r") as f:
            for line in f.readlines():
                if line.startswith("input_dimensionality="):
                    input_dimensionality = int(line[line.index("=") + 1:].strip())
                elif line.startswith("output_dimensionality="):
                    output_dimensionality = int(line[line.index("=") + 1:].strip())
                elif line.startswith("num_layers="):
                    num_layers = int(line[line.index("=") + 1:].strip())
                elif line.startswith("noise="):
                    noise = float(line[line.index("=") + 1:].strip())
                else:
                    raise ValueError("Invalid line: \"%s\"" % line)

        # make sure everything is there
        assert input_dimensionality is not None
        assert output_dimensionality is not None
        assert num_layers is not None
        assert noise is not None

        randomized_indices = np.load(filename_prefix + "_randidx.npy")

        reduction_layer_blocks = []
        num_blocks = int(np.ceil(float(len(randomized_indices))/output_dimensionality))
        for blockidx in range(num_blocks):
            reduction_layer_blocks.append(np.load(filename_prefix + "_block%s.npy" % blockidx))

        # load hidden layer W matrices
        layer_weights = []
        for layeridx in range(num_layers - 1):
            layer_weights.append(np.load(filename_prefix + "_layer%s.npy" % layeridx))

        msda = mSDA(noise, num_layers, input_dimensionality, output_dimensionality)

        msda.reduction_layer.randomized_indices = randomized_indices
        msda.reduction_layer.blocks = reduction_layer_blocks

        for layer_num, mda_layer in enumerate(msda.mda_layers):
            mda_layer.blocks.append(layer_weights[layer_num])

        return msda

