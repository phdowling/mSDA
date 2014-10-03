import numpy as np
import logging
ln = logging.getLogger("mDA")
ln.setLevel(logging.DEBUG)

from scipy import sparse
from scipy.sparse import vstack, csc_matrix, csr_matrix


class mDA(object):
    def __init__(self, noise, lambda_, weights=None, highdimen=False):
        self.noise = noise
        self.lambda_ = lambda_
        if not weights:
            self.weights = []
        else:
            self.weights = weights

        self.reduce_dimensionality = highdimen

    def train(self, input_data, return_hidden=False, reduced_representations=None):
        dimensionality, num_documents = input_data.shape
        output_dim = dimensionality

        if self.reduce_dimensionality:
            assert reduced_representations is not None
            reduced_dim, num_docs2 = reduced_representations.shape
            assert num_docs2 == num_documents

            output_dim = reduced_dim

        ln.debug("mDA beginning training.")

        bias = csc_matrix(np.ones((1, num_documents)))

        input_data = vstack((input_data, bias)).tocsc()

        scatter = input_data.dot(input_data.T)  # scatter is symmetric!

        corruption = csc_matrix(np.ones((dimensionality + 1, 1))) * (1 - self.noise)
        corruption[-1] = 1

        # this is a hacky translation of the original Matlab code, to avoid allocating a big (d+1)x(d+1) matrix
        # instead of element-wise multiplying the matrices, we handle the corresponding areas individually

        # corrupt everything
        Q = scatter * (1-self.noise)**2
        # partially undo corruption to values in (d+1,:)
        Q[dimensionality] = scatter[dimensionality] * (1.0/(1-self.noise))
        # partially undo corruption to values in (:,d+1)
        Q[:, dimensionality] = scatter[:, dimensionality] * (1.0/(1-self.noise))
        # undo corruption of (-1, -1)
        Q[-1, -1] = scatter[-1, -1] * (1.0/(1-self.noise)**2)

        # replace the diagonal (this is according to the original code again)
        idxs = range(dimensionality + 1)
        Q[idxs, idxs] = corruption.T.multiply(scatter[idxs, idxs])

        ln.debug("nnz for Q: %s" % (Q.nnz,))

        # Constructing P
        if self.reduce_dimensionality:
            P = reduced_representations.dot(input_data.T.tocsc()) * (1 - self.noise)
            P[:, dimensionality] *= (1.0 / (1 - self.noise))

        else:
            P = scatter.tocsr()[:-1, :].tocsc()
            P *= (1 - self.noise)
            P[:, dimensionality] *= (1.0 / (1 - self.noise))

        ln.debug("nnz for P: %s" % (P.nnz,))

        reg = sparse.eye(dimensionality + 1, format="csc").multiply(self.lambda_)
        reg[-1, -1] = 0

        # we need to solve W = P * Q^-1
        # Q is symmetric, so Q = Q^T
        # WQ = P
        # (WQ)^T = P^T
        # Q^T W^T = P^T
        # Q W^T = P^T

        # basically, self.weights = np.linalg.lstsq((Q + reg), P.T)[0].T

        Qreg = (Q + reg).todense()  # This is based on Q and reg, and is therefore symmetric
        PT = csc_matrix(P.T)
        PT.sort_indices()

        # Solving for W
        # W is going to be dx(d+1) (or rx(d+1) for high dimensions)
        self.weights = np.zeros((0, dimensionality + 1))

        if self.reduce_dimensionality:
            num_batches = 1
        else:
            # TODO choose a sensible value automatically
            num_batches = 3

        # we solve the system in batched to possibly conserve some memory for high dimensional data
        batch_size = int(np.ceil(float(output_dim) / num_batches))
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = int(min((batch_idx + 1) * batch_size, output_dim))
            column_idxs = range(start, end)

            # PT is (d+1) x r
            pt_columns = PT[:, column_idxs].todense()

            ln.debug("Solving (Q+reg)W^T = columns. Columns is %s by %s" % pt_columns.shape)
            weights = np.linalg.lstsq(Qreg, pt_columns)[0].T

            ln.debug("weights: %s, %s" % weights.shape)

            self.weights = np.vstack([self.weights, weights])

            ln.debug("finished batch %s" % (batch_idx,))

        ln.debug("nnz for weights: %s" % (np.count_nonzero(self.weights)))

        ln.debug("finished training.")

        del P
        del Q
        del scatter
        del bias
        del corruption
        
        if return_hidden:
            ln.debug("Computing and returning hidden representations.")

            if self.reduce_dimensionality:
                hidden_representations = self.weights.dot(input_data.todense())
            else:  # TODO: change this to sparse when we're not doing dimensional reduction
                hidden_representations = self.weights.dot(input_data.todense())
                hidden_representations = np.tanh(hidden_representations)
            del input_data
            ln.debug("nnz for hidden: %s" % (np.count_nonzero(hidden_representations)))
            return hidden_representations


    def get_hidden_representations(self, input_data):
        dimensionality, num_documents = input_data.shape

        bias = np.ones((1, num_documents))

        biased = np.concatenate((input_data, bias))

        biased = np.matrix(biased)

        hidden_representations = np.dot(self.weights, biased)
        if not self.reduce_dimensionality:
            hidden_representations = np.tanh(hidden_representations)

        del biased
        del bias

        return hidden_representations

