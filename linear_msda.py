import numpy as np
import logging
ln = logging.getLogger("mSDA")
ln.setLevel(logging.DEBUG)
from linear_mda import mDA

from scipy.sparse import csc_matrix, lil_matrix

from collections import defaultdict


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
        ln.debug("got %s input documents." % (len(input_data)))

        results = []

        ln.debug("Constructing sparse matrix for corpus.")
        acc = convert_to_sparse_matrix(input_data, self.input_dimensionality)
        #for idx, document in enumerate(input_data):
        #    acc.append(convert(document, self.input_dimensionality))
        #acc = np.concatenate(acc, axis=1)

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
        return self._msda.get_hidden_representations(input_data)


class mSDAhd(object):
    """
    (Linear) marginalized Stacked Denoising Autoencoder with dimensionality reduction, a.k.a. dense Cohort of Terms (dCoT).
    Use this class for creating semantic models of textual data. 
    """
    def __init__(self, prototype_ids, input_dimensionality, noise=0.5, num_layers=5):
        self._msda = _mSDA(noise, num_layers, highdimen=True)
        self.prototype_ids = prototype_ids
        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = len(prototype_ids)

    def train(self, input_data, return_hidden=False, streamed=False, batch_size=3000):
        #reduced_representations
        num_docs = len(input_data)
        if not streamed:
            batch_size = num_docs

        ln.debug("got %s input documents, batch size is %s" % (len(input_data), batch_size))

        batches = 0
        acc = []
        if return_hidden:
            results = []
        for idx, document in enumerate(input_data):
            acc.append(convert(document, self.input_dimensionality))

            if (idx % batch_size == 0 and idx != 0) or idx == num_docs:
                batches += 1
                acc = np.concatenate(acc, axis=1)
                representations = self._msda.train(acc, return_hidden=return_hidden, reduced_representations=acc[self.prototype_ids, :])
                if return_hidden:
                    for row in np.concatenate([rep.T for rep in representations], axis=1):
                        results.append(row)
                acc = []
                ln.debug("trained on %s documents, (meta-)batch number %s.." % (idx, batches))
        if acc:
            ln.error("This shouldn't happen, there's still something in the accumulator!")
        del acc

        if return_hidden:
            return results

    def get_hidden_representations(self, input_data):
        acc = np.concatenate([convert(document, self.input_dimensionality) for document in input_data], axis=1)
        reps = self._msda.get_hidden_representations(acc)

        return reps


class _mSDA(object):
    """
    Implementation class for both regular and dimensionality reduction mSDA. 
    Probably don't want to initialize this directly, the provided utility classes are easier to deal with.
    """
    def __init__(self, noise, num_layers, highdimen=False):
        self.highdimen = highdimen
        self.lambda_ = 1e-05
        self.noise = noise
        if self.highdimen:
            self.hdlayer = []
            self.layers = [mDA(noise, self.lambda_) for _ in range(num_layers - 1)]
            self.randomized_indices = None
            self.reduced_dim = None
        else:
            self.layers = [mDA(noise, self.lambda_) for _ in range(num_layers)]

    def train(self, input_data, return_hidden=False, reduced_representations=None):
        """
        train (or update) the underlying linear mapping. 

        if dimensional reduction should be performed (i.e. mSDA was initialized with highdimen=True), reduced_representations must 
        contain a projection into a lower-dimensional subspace of the word space. usually, the top k most frequent terms are chosen.
        """

        dimensionality, num_documents = input_data.shape
        ln.debug("dimensionality is %s, num_documents is %s" % (dimensionality, num_documents))
        
        # this handles the initial dimensional reduction
        if self.highdimen:
            ln.debug("High dimen is True")
            assert reduced_representations is not None
            reduced_dim, num_docs2 = reduced_representations.shape
            assert num_docs2 == num_documents

            if self.reduced_dim is None:
                self.reduced_dim = reduced_dim
            else:
                assert reduced_dim == self.reduced_dim

            current_representation = np.zeros((self.reduced_dim, num_documents))
            
            if self.randomized_indices is None:
                self.randomized_indices = np.random.permutation(dimensionality)

            ln.debug("Performing initial dimensional reduction with %s folds" % (dimensionality/self.reduced_dim))
            for batch in range(dimensionality/self.reduced_dim):
                indices = self.randomized_indices[batch*self.reduced_dim: (batch + 1)*self.reduced_dim]

                ln.debug("Initial dimensional reduction on fold %s.." % batch)

                if len(self.hdlayer) <= batch:
                    mda = mDA(self.noise, self.lambda_, highdimen=True)
                    self.hdlayer.append(mda)
                else:
                    mda = self.hdlayer[batch]

                hidden = mda.train(input_data[indices, :], return_hidden=True,
                                   reduced_representations=reduced_representations)
                current_representation += hidden
                del hidden

            current_representation /= len(range(dimensionality/self.reduced_dim))
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
            
            for batch in range(dimensionality/self.reduced_dim):
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
