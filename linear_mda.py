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


        ln.debug("Applying corrution vector to compute Q")
        #Q = scatter.multiply(corruption.dot(corruption.T))
        Q = csc_matrix((dimensionality + 1, dimensionality + 1))

        ln.debug("scatter: %s" % (scatter.__repr__()))

        ln.debug("...multiplying values up to (d,d)")
        Q = scatter * (1-self.noise)**2
        ln.debug("...multiplying values in (d+1,:)")
        Q[dimensionality] = scatter[dimensionality] * (1.0/(1-self.noise))
        ln.debug("...multiplying values in (:,d+1)")
        Q[:, dimensionality] = scatter[:, dimensionality] * (1.0/(1-self.noise))
        Q[-1, -1] = scatter[-1, -1] * (1.0/(1-self.noise)**2)

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
            ln.debug("converting to csr, slicing..")
            P = scatter.tocsr()[:-1, :].tocsc()
            #ln.debug("vstacking corruption")
            #corrupt = vstack(dimensionality * [corruption.T])
            ln.debug("P: %s, corruption: %s" % (repr(P), repr(corruption)))
            ln.debug("multiplying")
            P *= (1 - self.noise)
            P[:, dimensionality] *= (1.0 / (1 - self.noise))

        ln.debug("Constructing reg")
        reg = sparse.eye(dimensionality+1, format="csc").multiply(self.lambda_)
        reg[-1, -1] = 0

        # we need to solve W = P * Q^-1
        # Q is symmetric, so Q = Q^T
        # WQ = P
        # (WQ)^T = P^T
        # Q^T W^T = P^T
        # Q W^T = P^T

        #self.weights = np.linalg.lstsq((Q + reg), P.T)[0].T
        # TODO: we need to compute the least square solution for each column of P.T, then hstack the results
        self.weights = csc_matrix((0, dimensionality + 1))

        PT = csc_matrix(P.T)
        ln.debug("Solving for W")
        for column in range(dimensionality):
            if column % 500 == 0:
                ln.debug("on column %s" % (column))
            w_row = sparse.linalg.lsmr((Q + reg), PT[:, column].todense()).T

            self.weights = sparse.vstack(self.weights, w_row)
        ln.debug("finished training.")

        del P
        del Q
        del scatter
        del bias
        del corruption
        
        if return_hidden:
            ln.debug("Computing hidden representations..")
            hidden_representations = self.weights.dot(input_data[:, :-1])
            if not self.highdimen:
                hidden_representations = np.tanh(hidden_representations)
            del input_data
            ln.debug("done.")
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

