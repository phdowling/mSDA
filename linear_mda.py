import numpy as np
import logging
ln = logging.getLogger("mDA")
ln.setLevel(logging.DEBUG)

from scipy import sparse
from scipy.sparse import vstack
from scipy.sparse import csc_matrix

class mDA(object):
    def __init__(self, noise, lambda_, weights=None, highdimen=False):
        self.noise = noise
        self.lambda_ = lambda_
        if not weights:
            self.weights = []
        else:
            self.weights = weights

        self.highdimen = highdimen

    def train(self, input_data, return_hidden=False, reduced_representations=None):
        dimensionality, num_documents = input_data.shape
        if self.highdimen:
            assert reduced_representations is not None
            reduced_dim, num_docs2 = reduced_representations.shape
            assert num_docs2 == num_documents

        ln.debug("mDA is beginning training.")

        bias = csc_matrix(np.ones((1, num_documents)))

        input_data = vstack((input_data, bias))

        ln.debug("Created bias matrix, now computing scatter matrix")
        scatter = input_data.dot(input_data.T)

        corruption = csc_matrix(np.ones((dimensionality + 1, 1))) * (1 - self.noise)
        corruption[-1] = 1

        # This part is slow. Can we work around this?
        ln.debug("Applying corrution vector to compute Q")
        #Q = scatter.multiply(corruption.dot(corruption.T))
        Q = csc_matrix((dimensionality + 1, dimensionality + 1))

        Q[:dimensionality, :dimensionality] = scatter[:dimensionality, :dimensionality] * (1-self.noise)**2
        Q[dimensionality] = scatter[dimensionality] * (1-self.noise)
        Q[:, dimensionality] = scatter[:, dimensionality] * (1-self.noise)
        Q[-1, -1] = scatter[-1, -1]

        # replace the diagonal of Q
        ln.debug("Replacing Q's diagonal")
        idxs = range(dimensionality + 1)
        Q[idxs, idxs] = corruption.T.multiply(scatter[idxs, idxs])

        ln.debug("Constructing P")
        if self.highdimen:
            #TODO: this will be broken for now
            P = np.multiply(
                reduced_representations.dot(input_data.T).todense(),
                np.tile(corruption.T, (reduced_dim, 1))
            )
        else:
            P = scatter[:-1, :].multiply(
                vstack(dimensionality * [corruption.T])
            )

        ln.debug("Constructing reg")
        reg = csc_matrix.eye(dimensionality+1).multiply(self.lambda_)
        reg[-1, -1] = 0

        # we need to solve W = P * Q^-1
        # Q is symmetric, so Q = Q^T
        # WQ = P
        # (WQ)^T = P^T
        # Q^T W^T = P^T
        # Q W^T = P^T
        ln.debug("Solving for weights matrix")
        #self.weights = np.linalg.lstsq((Q + reg), P.T)[0].T
        self.weights = sparse.linalg.lsqr((Q + reg), P.T)[0].T
        ln.debug("finished training.")
        del P
        del Q
        del scatter
        del bias
        del corruption
        
        if return_hidden:
            hidden_representations = self.weights.dot(input_data[:, :-1])
            if not self.highdimen:
                hidden_representations = np.tanh(hidden_representations)
            del input_data
            return hidden_representations


    def get_hidden_representations(self, input_data):
        dimensionality, num_documents = input_data.shape

        bias = np.ones((1, num_documents))

        biased = np.concatenate((input_data, bias))

        biased = np.matrix(biased)

        hidden_representations = np.dot(self.weights, biased)
        if not self.highdimen:
            hidden_representations = np.tanh(hidden_representations)

        del biased
        del bias

        return hidden_representations

