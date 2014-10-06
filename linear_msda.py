import numpy as np
import logging
ln = logging.getLogger("mSDA")
ln.setLevel(logging.DEBUG)
from linear_mda import mDA

from scipy import sparse
from scipy.sparse import csc_matrix, lil_matrix

from ast import literal_eval

def convert(sparse_bow, dimensionality):
    dense = np.zeros((dimensionality, 1))
    for dim, value in sparse_bow:
        dense[dim] = value
    return csc_matrix(dense)

def convert_to_sparse_matrix(input_data, dimensionality):
    sparse = lil_matrix((dimensionality, len(input_data)))
    for docidx, document in enumerate(input_data):
        if docidx % 5000 == 0:
            ln.debug("on document %s.." % (docidx,))
        for word_id, count in document:
            sparse[word_id, docidx] = count
    return sparse.tocsc()

class mSDA(object):
    """
    (Linear) marginalized Stacked Denoising Autoencoder. 
    """
    def __init__(self, noise, num_layers, input_dimensionality):
        self._msda = _mSDA(noise, num_layers, highdimen=False)
        self.input_dimensionality = input_dimensionality

    def train(self, input_data, return_hidden=False):
        """
        input_data must be a numpy array, where each row represents one training documents/image/etc.
        """

        results = []

        ln.debug("Constructing sparse matrix for corpus.")
        acc = convert_to_sparse_matrix(input_data, self.input_dimensionality)

        ln.debug("Beginning mSDA training.")
        representations = self._msda.train(acc, return_hidden=return_hidden)
        if return_hidden:
            for row in np.concatenate([rep.T for rep in representations], axis=1):
                results.append(row)
        del acc
        del representations

        ln.debug("Training done.")

        if return_hidden:
            return results

    def get_hidden_representations(self, input_data):
        acc = convert_to_sparse_matrix(input_data, self.input_dimensionality)
        return self._msda.get_hidden_representations(acc)

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
    def __init__(self, prototype_ids, input_dimensionality, noise=0.5, num_layers=5):
        self._msda = _mSDA(noise, num_layers, highdimen=True, reduced_dim=len(prototype_ids))
        self.prototype_ids = prototype_ids
        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = len(prototype_ids)

    def train(self, input_data, return_hidden=False):
        #reduced_representations
        num_docs = len(input_data)
        ln.debug("got %s input documents." % num_docs)

        results = []

        ln.debug("Constructing sparse matrix for corpus.")
        acc = convert_to_sparse_matrix(input_data, self.input_dimensionality)

        ln.debug("Beginning mSDA training.")

        representations = self._msda.train(acc, return_hidden=return_hidden,
                                           reduced_representations=acc[self.prototype_ids, :])
        if return_hidden:
            for row in np.concatenate([rep.T for rep in representations], axis=1):
                results.append(row)
        del acc
        del representations

        ln.debug("Training done.")

        if return_hidden:
            return results


    def get_hidden_representations(self, input_data):
        acc = convert_to_sparse_matrix(input_data, self.input_dimensionality)
        reps = self._msda.get_hidden_representations(acc)

        return reps

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
    def __init__(self, noise, num_layers, highdimen=False, reduced_dim=None):
        self.highdimen = highdimen
        self.lambda_ = 1e-05
        self.noise = noise
        if self.highdimen:
            self.hdlayer = []
            self.layers = [mDA(noise, self.lambda_) for _ in range(num_layers - 1)]
            self.randomized_indices = None
            self.reduced_dim = reduced_dim
        else:
            self.layers = [mDA(noise, self.lambda_) for _ in range(num_layers)]

    def train(self, input_data, return_hidden=False, reduced_representations=None):
        """
        train (or update) the underlying linear mapping. 

        if dimensional reduction should be performed (i.e. mSDA was initialized with highdimen=True), reduced_representations must 
        contain a projection into a lower-dimensional subspace of the word space. usually, the top k most frequent terms are chosen.
        """

        dimensionality, num_documents = input_data.shape
        
        # this handles the initial dimensional reduction
        if self.highdimen:
            assert reduced_representations is not None
            reduced_dim, num_docs2 = reduced_representations.shape
            assert num_docs2 == num_documents
            assert reduced_dim == self.reduced_dim

            current_representation = csc_matrix((self.reduced_dim, num_documents))
            
            if self.randomized_indices is None:
                self.randomized_indices = np.random.permutation(dimensionality)

            ln.debug("Performing initial dimensional reduction with %s folds" % (int(np.ceil(float(dimensionality) /
                                                                                             self.reduced_dim))))
            for batch in range(int(np.ceil(float(dimensionality)/self.reduced_dim))):
                indices = self.randomized_indices[batch*self.reduced_dim: (batch + 1)*self.reduced_dim]

                ln.debug("Initial dimensional reduction on fold %s.." % batch)

                if len(self.hdlayer) <= batch:
                    mda = mDA(self.noise, self.lambda_, highdimen=True)
                    self.hdlayer.append(mda)
                else:
                    mda = self.hdlayer[batch]

                hidden = mda.train(input_data[indices, :], return_hidden=True,
                                   reduced_representations=reduced_representations)
                current_representation = current_representation + hidden
                del hidden

            current_representation = (1.0 / len(range(dimensionality/self.reduced_dim))) * current_representation
            current_representation = np.tanh(current_representation)
        else:
            current_representation = input_data

        if return_hidden:
            representations = [current_representation]
        ln.debug("Now training %s layers." % (len(self.layers),))
        for idx, layer in enumerate(self.layers):
            current_representation = layer.train(current_representation, return_hidden=True)
            if return_hidden:
                representations.append(current_representation)
            ln.debug("trained layer %s" % (idx+1,))

        if return_hidden:
            return representations

    def get_hidden_representations(self, input_data):
        """
        convert 
        """
        dimensionality, num_documents = input_data.shape
        if self.highdimen:
            assert self.randomized_indices is not None
            current_representation = np.zeros((self.reduced_dim, num_documents))    
            
            for batch in range(int(np.ceil(float(dimensionality)/self.reduced_dim))):
                indices = self.randomized_indices[batch*self.reduced_dim: (batch + 1)*self.reduced_dim]
                
                mda = self.hdlayer[batch]

                hidden = mda.get_hidden_representations(input_data[indices, :])
                
                current_representation += hidden
            current_representation = current_representation / np.ceil(dimensionality/self.reduced_dim)
            current_representation = np.tanh(current_representation)
        else:
            current_representation = input_data

        representations = [current_representation]
        for layer in self.layers:
            current_representation = layer.get_hidden_representations(current_representation)
            representations.append(current_representation)

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

        noise = None
        with open(filename_prefix, "r") as f:
            for line in f.readlines():
                if line.startswith("highdimen="):
                    highdimen = line.strip().endswith("True")
                elif line.startswith("num_layers="):
                    num_layers = int(line[line.index("=") + 1:].strip())
                elif line.startswith("noise="):
                    noise = float(line[line.index("=") + 1:].strip())
                else:
                    raise ValueError("Invalid line: \"%s\"" % line)

        # make sure everything is there
        assert highdimen is not None
        assert num_layers is not None
        assert noise is not None
        print highdimen
        print num_layers

        # load hidden layer W matrices
        layers = []
        for layeridx in range(num_layers - (1 if highdimen else 0)):
            layers.append(np.load(filename_prefix + "_layer%s.npy" % layeridx))

        # if HD: load the W matrices for each block in the first layer
        blocks = []
        if highdimen:
            randomized_indices = np.load(filename_prefix + "_randidx.npy")

            output_dim, _ = layers[0].shape
            num_blocks = int(np.ceil(float(len(randomized_indices))/output_dim))
            for blockidx in range(num_blocks):
                blocks.append(np.load(filename_prefix + "_block%s.npy" % blockidx))

        # initialize mSDA
        if highdimen:
            msda = mSDAhd(prototype_ids=range(output_dim), input_dimensionality=len(randomized_indices), noise=noise,
                          num_layers=num_layers)
            for block in blocks:
                lambda_ = 1e-05
                msda._msda.hdlayer.append(mDA(noise=noise, lambda_=lambda_, weights=block, highdimen=True))
            msda._msda.randomized_indices = randomized_indices
        else:
            msda = mSDA(input_dimensionality=len(randomized_indices), noise=noise, num_layers=num_layers)

        # assign layers with the loaded W matrices
        for idx, layer in enumerate(msda._msda.layers):
            layer.weights = layers[idx]

        return msda
