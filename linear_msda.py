__author__ = 'dowling'
import logging
import numpy as np

ln = logging.getLogger("mSDA")
ln.setLevel(logging.DEBUG)

from models.impl.msda.linear_mda import mDA

from models.impl.gensim import utils, matutils
from scipy.sparse import csc_matrix, lil_matrix

from tempfile import TemporaryFile

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


class FilteringChunkDocumentStream(object):
    def __init__(self, corpus, num_terms, filter_dimensions=None, chunksize=5000):
        self.corpus = corpus
        self.num_terms = num_terms
        self.filter_dimensions = filter_dimensions
        self.chunksize = chunksize

    def __iter__(self):
        for chunk_no, chunk in enumerate(utils.grouper(self.corpus, self.chunksize)):
            # ln.info("preparing a new chunk of documents")
            nnz = sum(len(doc) for doc in chunk)
            # construct the job as a sparse matrix, to minimize memory overhead
            # definitely avoid materializing it as a dense matrix!
            # ln.debug("converting corpus to csc format")
            job = matutils.corpus2csc(chunk, num_docs=len(chunk), num_terms=self.num_terms, num_nnz=nnz)

            if self.filter_dimensions is not None:
                job = job[self.filter_dimensions, :]

            yield job
            del chunk


class TempFileStream(object):
    def __init__(self, tempfiles):
        self.tempfiles = tempfiles

    def __iter__(self):
        for temp_file in self.tempfiles:
            temp_file.seek(0)
            matrix = np.load(temp_file)
            yield matrix
            del matrix

class mSDA(object):
    """
    (Linear) marginalized Stacked Denoising Autoencoder.
    """
    def __init__(self, noise, num_layers, input_dimensionality):
        self._msda = _mSDA(noise, num_layers, input_dimensionality)
        self.input_dimensionality = input_dimensionality

    def train(self, input_data, chunksize=5000):
        """
        input_data must be a numpy array, where each column represents one training documents/image/etc.
        """
        ln.debug("Beginning mSDA training.")
        self._msda.train(input_data)

        ln.debug("Training done.")

    def get_hidden_representations(self, input_data):
        acc = convert_to_sparse_matrix(input_data, self.input_dimensionality)
        return self._msda.get_hidden_representations(acc)

    def __getitem__(self, bow):
        is_corpus, corpus = utils.is_corpus(bow)
        if is_corpus:
            return self.get_hidden_representations(bow)
        else:
            return self.get_hidden_representations([bow])

    def save(self, filename):
        self._msda.save(filename)

    @classmethod
    def load(cls, filename):
        return _mSDA.load(filename)


class mSDAhd(object):
    """
    (Linear) marginalized Stacked Denoising Autoencoder with dimensionality reduction, a.k.a. dense Cohort of Terms (dCoT).
    Use this class for creating semantic models of textual data.
    """
    def __init__(self, dimensions, id2word, noise=0.5, num_layers=5, input_dim=None):
        if id2word is None:
            assert input_dim
            ln.warn("Initialized mSDAhd without id2word - not selecting prototype ids, no training possible.")
            self._msda = _mSDA(noise, num_layers, input_dim, output_dimensionality=dimensions)
            self.prototype_ids = None
            self.input_dimensionality = input_dim
        else:
            self._msda = _mSDA(noise, num_layers, len(id2word), output_dimensionality=dimensions)
            self.prototype_ids = [id_ for id_, val in
                              sorted(id2word.items(), key=(lambda (k, v): -id2word.dfs[k]))][:dimensions]
            self.input_dimensionality = len(id2word)
        self.output_dimensionality = dimensions

    def train(self, input_data, chunksize=10000):
        assert self.prototype_ids
        #ln.debug("Constructing sparse matrix for corpus.")
        #acc = convert_to_sparse_matrix(input_data, self.input_dimensionality)
        filtered_doc_chunks = FilteringChunkDocumentStream(input_data, self.input_dimensionality,
                                                           self.prototype_ids, chunksize=chunksize)

        ln.debug("Beginning mSDA training.")

        self._msda.train(input_data,  reduced_representations=filtered_doc_chunks, chunksize=chunksize)

        ln.debug("Training done.")

    def get_hidden_representations(self, input_data, seperate=False):
        acc = convert_to_sparse_matrix(input_data, self.input_dimensionality)
        reps = self._msda.get_hidden_representations(acc, seperate=seperate)

        return reps

    def __getitem__(self, bow):
        is_corpus, corpus = utils.is_corpus(bow)
        if is_corpus:
            return self.get_hidden_representations(bow, seperate=True)
        else:
            return self.get_hidden_representations([bow], seperate=True)[0]

    def save(self, filename):
        self._msda.save(filename)

    @classmethod
    def load(cls, filename):
        return _mSDA.load(filename)


class _mSDA(object):
    """
    Implementation class for both regular and dimensionality reduction mSDA.
    Probably don't want to initialize this directly, the provided utility classes are easier to deal with.
    """
    def __init__(self, noise, num_layers, input_dimensionality, output_dimensionality=None):
        self.highdimen = bool(output_dimensionality)
        self.lambda_ = 1e-05
        self.noise = noise
        self.input_dim = input_dimensionality
        if self.highdimen:
            self.hdlayer = []
            self.layers = [mDA(noise, self.lambda_, output_dimensionality) for _ in range(num_layers-1)]
            self.randomized_indices = None
            self.output_dim = output_dimensionality
        else:
            self.layers = [mDA(noise, self.lambda_, input_dimensionality) for _ in range(num_layers)]

    def train(self, input_data, reduced_representations=None, chunksize=5000):
        """
        train (or update) the underlying linear mapping.

        if dimensional reduction should be performed (i.e. mSDA was initialized with highdimen=True), reduced_representations must
        contain a projection into a lower-dimensional subspace of the word space. usually, the top k most frequent terms are chosen.
        """
        # input_data is a gensim corpus compatible format

        current_representation_chunks = []

        # this handles the initial dimensional reduction
        if self.highdimen:
            assert reduced_representations is not None

            if self.randomized_indices is None:
                self.randomized_indices = np.random.permutation(self.input_dim)

            ln.debug("Performing initial dimensional reduction with %s folds" % (
                int(np.ceil(float(self.input_dim) / self.output_dim))))

            for batch in range(int(np.ceil(float(self.input_dim)/self.output_dim))):
                indices = self.randomized_indices[batch*self.output_dim: (batch + 1)*self.output_dim]

                ln.debug("Initial dimensional reduction on fold %s.." % batch)

                if len(self.hdlayer) <= batch:

                    mda = mDA(self.noise, self.lambda_, input_dimensionality=len(indices),
                              output_dimensionality=self.output_dim)
                    self.hdlayer.append(mda)
                else:
                    mda = self.hdlayer[batch]

                indices_filtered_input = FilteringChunkDocumentStream(input_data, self.input_dim, indices, chunksize=chunksize)

                mda.train(indices_filtered_input, reduced_representations=reduced_representations)

                def convert_chunk_and_update_average(chunk_, chunk_no_):
                    if batch == 0:
                        temp_file = TemporaryFile()
                        current_representation_chunks.append(temp_file)
                        ln.debug("created new tempfile, initializing avg chunk. first index batch, corpus chunk %s" % chunk_no)
                        running_average = mda[chunk_]
                    else:
                        temp_file = current_representation_chunks[chunk_no_]
                        temp_file.seek(0)
                        running_average = np.load(temp_file)
                        hidden = mda[chunk_]
                        running_average += (1.0 / batch + 1) * (hidden - running_average)
                    np.save(temp_file, running_average)
                    ln.debug("saved updated avg, corpus chunk %s." % chunk_no)

                for chunk_no, chunk in enumerate(indices_filtered_input):
                    convert_chunk_and_update_average(chunk, chunk_no)

            # apply tanh element wise
            ln.debug("Applying tanh to reduction layer chunks..")
            for chunk_file in current_representation_chunks:
                chunk_file.seek(0)
                running = np.load(chunk_file)
                running = np.tanh(running)
                np.save(chunk_file, running)

            current_representation = TempFileStream(current_representation_chunks)

        else:
            current_representation = FilteringChunkDocumentStream(input_data, self.input_dim, chunksize=chunksize)

        ln.debug("Now training %s layers." % (len(self.layers),))
        for layer_num, layer in enumerate(self.layers):
            layer.train(current_representation)

            # we compute the hidden representation for each chunk and overwrite its previous representation
            def convert_chunk_and_replace_representation(chunk_, chunk_no_):
                if not self.highdimen and layer_num == 0:
                    current_representation_chunks.append(TemporaryFile())
                temp_file = current_representation_chunks[chunk_no_]
                temp_file.seek(0)
                hidden = layer[chunk_]
                np.save(temp_file, hidden)

            for chunk_no, chunk in enumerate(current_representation):
                convert_chunk_and_replace_representation(chunk, chunk_no)

            if not self.highdimen and layer_num == 0:
                current_representation = TempFileStream(current_representation_chunks)

            ln.debug("trained layer %s" % (layer_num + 1,))

        del current_representation
        for representation_file in current_representation_chunks:
            representation_file.close()
        ln.info("mSDA finished training.")

    def get_hidden_representations(self, input_data, seperate=True, concatenate=True):
        """
        convert a sparse matrix of documents to their mSDA representation.
        if seperate is true, return the documents in list form
        if concatenate is true, the representation of each document is the concatenation of each layer output
            otherwise, use the last layers' output only
        """
        dimensionality, num_documents = input_data.shape
        if self.highdimen:
            assert self.randomized_indices is not None
            current_representation = np.zeros((self.output_dim, num_documents))

            for batch in range(int(np.ceil(float(dimensionality)/self.output_dim))):
                indices = self.randomized_indices[batch*self.output_dim: (batch + 1)*self.output_dim]

                mda = self.hdlayer[batch]

                hidden = mda[input_data[indices, :]]

                # running average
                current_representation += (1.0 / batch + 1) * (hidden - current_representation)

            #current_representation = current_representation / np.ceil(dimensionality/self.output_dim)
            current_representation = np.tanh(current_representation)
        else:
            current_representation = input_data

        representations = None
        if concatenate:
            representations = [current_representation]

        for layer in self.layers:
            current_representation = layer.get_hidden_representations(current_representation)
            if concatenate:
                representations.append(current_representation)

        if concatenate:
            representations = np.vstack(representations)
        else:
            representations = current_representation

        if seperate:
            return matutils.Dense2Corpus([column for column in representations.T])
        else:
            return representations



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
            if self.highdimen:
                f.write("highdimen=True\n")
                f.write("num_layers=%s\n" % (len(self.layers) + 1,))
            else:
                f.write("highdimen=False\n")
                f.write("num_layers=%s\n" % (len(self.layers),))
            f.write("noise=%s" % self.noise)
            f.write("input_dim=%s" % self.input_dim)

        if self.highdimen:
            np.save(filename_prefix + "_randidx", self.randomized_indices)
            for idx, block in enumerate(self.hdlayer):
                np.save(filename_prefix + "_block%s" % idx, block.weights)
        for idx, layer in enumerate(self.layers):
            np.save(filename_prefix + "_layer%s" % idx, layer.weights)

    @classmethod
    def load(cls, filename_prefix):
        # load metadata
        highdimen = None
        num_layers = None
        input_dim = None

        noise = None
        with open(filename_prefix, "r") as f:
            for line in f.readlines():
                if line.startswith("highdimen="):
                    highdimen = line.strip().endswith("True")
                elif line.startswith("num_layers="):
                    num_layers = int(line[line.index("=") + 1:].strip())
                elif line.startswith("noise="):
                    noise = float(line[line.index("=") + 1:].strip())
                elif line.startswith("input_dim="):
                    input_dim = int(line[line.index("=") + 1:].strip())
                else:
                    raise ValueError("Invalid line: \"%s\"" % line)

        # make sure everything is there
        assert highdimen is not None
        assert num_layers is not None
        assert noise is not None

        # load hidden layer W matrices
        layers = []
        for layeridx in range(num_layers - (1 if highdimen else 0)):
            layers.append(np.load(filename_prefix + "_layer%s.npy" % layeridx))

        # if HD: load the W matrices for each block in the first layer
        blocks = []
        randomized_indices = None
        if highdimen:
            randomized_indices = np.load(filename_prefix + "_randidx.npy")

            output_dim, _ = layers[0].shape
            num_blocks = int(np.ceil(float(len(randomized_indices))/output_dim))
            for blockidx in range(num_blocks):
                blocks.append(np.load(filename_prefix + "_block%s.npy" % blockidx))

        # initialize mSDA
        if highdimen:
            msda = mSDAhd(output_dim, None, noise=noise, num_layers=num_layers, input_dim=input_dim)

            for block in blocks:
                lambda_ = 1e-05
                mda = mDA(noise=noise, lambda_=lambda_, input_dimensionality=input_dim, output_dimensionality=output_dim)
                mda.weights = block
                msda._msda.hdlayer.append(mda)
            msda._msda.randomized_indices = randomized_indices
        else:

            msda = mSDA(noise=noise, num_layers=num_layers, input_dimensionality=input_dim)

        # assign layers with the loaded W matrices
        for idx, layer in enumerate(msda._msda.layers):
            layer.weights = layers[idx]

        return msda

