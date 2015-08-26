__author__ = 'dowling'
import logging
import numpy as np

ln = logging.getLogger("mSDA")
ln.setLevel(logging.DEBUG)

from mda_layer import mDALayer

from gensim import utils, matutils

from scipy.sparse import csc_matrix, lil_matrix

from gensim.corpora.mmcorpus import MmCorpus

import os

#USE_MMCORPUS = False

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


class NumpyChunkCorpus(object):

    @staticmethod
    def serialize(filename_prefix, layer, current_representation, num_terms=None, chunksize=10000):
        is_corpus, current_representation = utils.is_corpus(current_representation)
        if is_corpus:
            for chunk_no, chunk in enumerate(utils.grouper(current_representation, chunksize)):
                ln.debug("preparing chunk for conversion (%s documents)..." % len(chunk))
                assert num_terms is not None, "Need num_terms to properly handle sparse corpus format"
                chunk_as_csc = matutils.corpus2csc(chunk, num_terms=num_terms)

                ln.debug("Chunk converted to csc, running through layer..")
                chunk_trans = layer.__getitem__(chunk_as_csc)

                ln.debug("Serializing hidden representation..")
                fname = "%s_%s" % (filename_prefix, ("0" * (15 - len(str(chunk_no)))) + str(chunk_no))
                np.save(fname, chunk_trans)
                ln.debug("Finished serializing chunk. Processed %s documents so far." %
                         (chunk_no * chunksize + len(chunk)))
        else:
            ln.info("Beginning serialization of non-gensim corpus format intermediate representation.")
            ln.debug("Type of current_representation is %s" % type(current_representation))
            for chunk_no, chunk in enumerate(current_representation):
                ln.debug("converting chunk (%s documents)..." % chunksize)
                chunk_trans = layer.__getitem__(chunk)
                ln.debug("Serializing hidden representation..")
                fname = "%s_%s" % (filename_prefix, ("0" * (15 - len(str(chunk_no)))) + str(chunk_no))
                np.save(fname, chunk_trans)
                ln.debug("finished serializing chunk.")

        ln.info("Finished serializing all chunks.")

    @staticmethod
    def load(filename_prefix):
        ln.debug("Loading intermediate representations..")
        filenames = []
        for filename in os.listdir(os.getcwd()):
            if filename.startswith(filename_prefix):
                filenames.append(filename)

        filenames.sort()
        ln.debug("Found %s files" % len(filenames))

        class chunk_iterator(object):
            def __init__(self, fnames):
                self.fnames = fnames

            def __iter__(self):
                for fname in self.fnames:
                    ln.debug("Loading file %s" % fname)
                    yield np.load(fname)

        return chunk_iterator(filenames)

    @staticmethod
    def cleanup(filename_prefix):
        for filename in os.listdir(os.getcwd()):
            if filename.startswith(filename_prefix):
                os.remove(filename)


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

        reduction_layer = mDALayer(noise=self.noise, lambda_=self.lambda_,
                                   input_dimensionality=self.input_dimensionality,
                                   output_dimensionality=self.output_dimensionality,
                                   prototype_ids=prototype_ids)

        self.reduction_layer = reduction_layer
        self.mda_layers = [mDALayer(noise, self.lambda_, input_dimensionality=self.output_dimensionality)
                           for _ in range(num_layers - 1)]

    @staticmethod
    def _save_intermediate(layer, current_representation, num_terms=None, chunksize=10000):
        #ln.debug("save_intermediate: %s" % chunksize)
        #if USE_MMCORPUS:
        #    ln.debug("calling __getitem__")
        #    transformed = layer.__getitem__(current_representation, chunksize=chunksize)
        #    ln.debug("serializing corpus")
        #    MmCorpus.serialize(".msda_intermediate.mm", transformed, progress_cnt=chunksize)
        #else:
        ln.debug("_save_intermediate called")
        NumpyChunkCorpus.serialize(".msda_intermediate", layer, current_representation=current_representation,
                                   num_terms=num_terms, chunksize=chunksize)


    @staticmethod
    def _load_intermediate():
        #if USE_MMCORPUS:
        #    return MmCorpus(".msda_intermediate.mm")
        #else:
        ln.debug("_load_intermediate called")
        return NumpyChunkCorpus.load(".msda_intermediate")

    @staticmethod
    def _cleanup_intermediate():
        #if USE_MMCORPUS:
        #    os.remove(".msda_intermediate.mm")
        #    os.remove(".msda_intermediate.mm.index")
        #else:
        ln.debug("_cleanup_intermediate called")
        NumpyChunkCorpus.cleanup(".msda_intermediate")

    def train(self, corpus, chunksize=10000):
        """
        train the underlying linear mappings.

        @param corpus is a gensim corpus compatible format
        @param use_temp_files determines whether to use temporary files to store the intermediate representations of
        the corpus to train the next layer. Setting flag True will not greatly affect memory usage, but will temporarily
        require a significant amount of disk space. Using temp files will strongly speed up training, especially as the
        number of layers increases.
        """
        #ln.debug("train: %s" % chunksize)
        ln.info("Training mSDA with %s layers.", len(self.mda_layers) + 1)
        #if not use_temp_files:
        #    ln.warn("Training without temporary files. May take a long time!")
        #    self.reduction_layer.train(corpus)
        #    current_representation = self.reduction_layer.__getitem__(corpus, chunksize=chunksize)
        #
        #    for layer_num, layer in enumerate(self.mda_layers):
        #
        #        # We feed the corpus through all intermediate layers to get the current representation
        #        # that representation is then used to train the next layer
        #        # this is memory-independent, but will probably be very slow.
        #
        #        ln.info("Training layer %s.", layer_num)
        #        layer.train(current_representation, chunksize=chunksize)
        #        if layer_num < len(self.mda_layers) - 1:
        #            current_representation = layer[current_representation]

       # else:
        ln.info("Using temporary files to speed up training.")

        ln.info("Beginning training on %s layers." % (len(self.mda_layers) + 1))
        self.reduction_layer.train(corpus, numpy_chunk_input=False, chunksize=chunksize)

        # serialize intermediate representation, load again (streamed) to train next layer

        self._save_intermediate(layer=self.reduction_layer, current_representation=corpus,
                                num_terms=self.input_dimensionality, chunksize=chunksize)
        current_representation = self._load_intermediate()

        for layer_num, layer in enumerate(self.mda_layers):
            layer.train(current_representation, numpy_chunk_input=True, chunksize=chunksize)
            if layer_num < len(self.mda_layers):
                self._save_intermediate(layer, current_representation, chunksize=chunksize)
                current_representation = self._load_intermediate()
        self._cleanup_intermediate()

        ln.info("mSDA finished training.")

    def _get_hidden_representations(self, input_data):
        """
        convert a numpy matrix of documents to their mSDA representation.
        if return_sparse is true, return the documents in list form, otherwise return a dense matrix
        if concatenate is true, the representation of each document is the concatenation of each layer output
            otherwise, use the last layers' output only
        """
        hidden = self.reduction_layer.__getitem__(input_data)
        for layer in self.mda_layers:
            hidden = layer.__getitem__(hidden)
        return hidden

    def __getitem__(self, bow, chunksize=10000):
        #ln.debug("getitem: %s" % chunksize)
        is_corpus, bow = utils.is_corpus(bow)
        if not is_corpus:
            bow = [bow]

        ln.info("Computing hidden representation for %s documents..." % len(bow))

        if not chunksize:  # todo I think could be removed altogether
            chunksize = 1

        def transformed_corpus():
            for chunk_no, doc_chunk in enumerate(utils.grouper(bow, chunksize)):
                ln.debug("Converting chunk %s to csc format.." % chunk_no)
                chunk = matutils.corpus2csc(doc_chunk, self.input_dimensionality)
                ln.debug("Computing hidden representation for chunk.. ")
                hidden = self._get_hidden_representations(chunk)
                ln.info("Finished computing representation for chunk %s, yielding results. Total docs processed: %s" %
                        (chunk_no, chunk_no * chunksize + len(doc_chunk)))
                for column in hidden.T:
                    yield matutils.dense2vec(column.T)
                ln.debug("Done yielding chunk %s" % chunk_no)

            ln.info("Finished computing representations for all chunks.")

        if not is_corpus:
            res = list(transformed_corpus()).pop()
            return res
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
        num_blocks = len(randomized_indices)
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

