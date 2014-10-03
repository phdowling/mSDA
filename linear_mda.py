import numpy as np
import logging
ln = logging.getLogger("mDA")
ln.setLevel(logging.DEBUG)

from scipy import sparse
from scipy.sparse import vstack
from scipy.sparse import csc_matrix, csr_matrix

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

        input_data = vstack((input_data, bias)).tocsc()

        #ln.debug("Created bias matrix, now computing scatter matrix")
        scatter = input_data.dot(input_data.T)
        # scatter is symmetric!

        corruption = csc_matrix(np.ones((dimensionality + 1, 1))) * (1 - self.noise)
        corruption[-1] = 1


        #ln.debug("Applying corrution vector to compute Q")
        # Q = scatter.multiply(corruption.dot(corruption.T))
        Q = csc_matrix((dimensionality + 1, dimensionality + 1))

        ln.debug("scatter: %s" % (scatter.__repr__()))

        #ln.debug("...multiplying values up to (d,d)")
        Q = scatter * (1-self.noise)**2
        #ln.debug("...multiplying values in (d+1,:)")
        Q[dimensionality] = scatter[dimensionality] * (1.0/(1-self.noise))
        #ln.debug("...multiplying values in (:,d+1)")
        Q[:, dimensionality] = scatter[:, dimensionality] * (1.0/(1-self.noise))
        Q[-1, -1] = scatter[-1, -1] * (1.0/(1-self.noise)**2)

        # replace the diagonal of Q
        #ln.debug("Replacing Q's diagonal")
        idxs = range(dimensionality + 1)
        Q[idxs, idxs] = corruption.T.multiply(scatter[idxs, idxs])

        #ln.debug("Constructing P")
        if self.highdimen:
            # P = xfreq*xxb'.*repmat(q', r, 1); ends up being rx(d+1)

            #input is now n*(d+1)
            #reduced_reps are r*d
            ln.debug("reduced: %s, input: %s" % (repr(reduced_representations), repr(input_data)))
            P = reduced_representations.dot(input_data.T.tocsc()) * (1 - self.noise)
            P[:, dimensionality] *= (1.0 / (1 - self.noise))


            #P = np.multiply(
            #    np.dot(reduced_representations, biased.T),
            #    np.tile(corruption.T, (reduced_dim, 1))
            #)

        else:
            #ln.debug("converting to csr, slicing..")
            P = scatter.tocsr()[:-1, :].tocsc()
            #ln.debug("vstacking corruption")
            #corrupt = vstack(dimensionality * [corruption.T])
            #ln.debug("P: %s, corruption: %s" % (repr(P), repr(corruption)))
            #ln.debug("multiplying")
            P *= (1 - self.noise)
            P[:, dimensionality] *= (1.0 / (1 - self.noise))

        #ln.debug("Constructing reg")
        reg = sparse.eye(dimensionality + 1, format="csc").multiply(self.lambda_)
        reg[-1, -1] = 0

        # we need to solve W = P * Q^-1
        # Q is symmetric, so Q = Q^T
        # WQ = P
        # (WQ)^T = P^T
        # Q^T W^T = P^T
        # Q W^T = P^T

        #self.weights = np.linalg.lstsq((Q + reg), P.T)[0].T

        Qreg = (Q + reg).todense()
        # This is based on Q and reg, and is therefore symmetric
        #ln.debug("Qreg: %s, %s" % Qreg.shape)
        PT = csc_matrix(P.T)
        PT.sort_indices()
        #ln.debug("Solving for W")
        #self.weights = sparse.linalg.spsolve(tosolve.tocsc(), PT)

        # gonna be dx(d+1)
        self.weights = np.zeros((reduced_dim, 0))

        if self.highdimen:
            num_batches = 1
        else:
            num_batches = 10

        batch_size = int(np.ceil(float(reduced_dim) / num_batches))
        for batch_idx in range(num_batches):
            #ln.debug("extracting columns..")
            start = batch_idx * batch_size
            end = int(min((batch_idx + 1) * batch_size, reduced_dim))
            column_idxs = range(start, end)
            # PT is (d+1) x r
            columns = PT[:, column_idxs].todense()
            ln.debug("Solving (Q+reg)W^T = columns. Columns is %s by %s" % columns.shape)
            weights = np.linalg.lstsq(Qreg, columns)[0].T
            ln.debug("weights: %s, %s" % weights.shape)

            self.weights = np.hstack([self.weights, weights])

            ln.debug("finished batch %s" % (batch_idx))
            ln.debug("%s" % repr(csc_matrix(weights)))

        ln.debug("finished training.")

        del P
        del Q
        del scatter
        del bias
        del corruption
        
        if return_hidden:
            ln.debug("Computing hidden representations..")
            # TODO: change this to sparse when we're not doing dimensional reduction
            hidden_representations = self.weights.dot(input_data.todense())
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

