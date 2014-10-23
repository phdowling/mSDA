__author__ = 'dowling'

import numpy as np
import logging
ln = logging.getLogger("mDA")
ln.setLevel(logging.DEBUG)

from scipy import sparse
from scipy.sparse import hstack, vstack, csc_matrix, csr_matrix

from itertools import izip


class mDA(object):
    def __init__(self, noise, lambda_, input_dimensionality, output_dimensionality=None):
        self.noise = noise
        self.lambda_ = lambda_
        self.input_dimensionality = input_dimensionality
        self.reduce_dimensionality = bool(output_dimensionality)
        if output_dimensionality:
            self.output_dimensionality = output_dimensionality

        self.weights = None

    def train(self, input_data, reduced_representations=None):
        ln.info("mDA beginning training.")

        # DIMENSIONS OVERVIEW
        # scatter: always (d+1) x (d+1)
        # P: in normal mDA d x (d+1), else r x (d+1)
        # Q: same as scatter
        # W: in normal mDA d x (d+1), else r x (d+1)

        # we do the following in limited memory with a streamed corpus to more documents

        scatter = np.zeros((self.input_dimensionality + 1, self.input_dimensionality + 1), dtype=float)
        if self.reduce_dimensionality:
            ln.debug("Input dim: %s. Output dim: %s" % (self.input_dimensionality, self.output_dimensionality))
            # the loop below is equivalent to the following:
            # P = reduced_representations.dot(input_data.T.tocsc())
            # P *= (1 - self.noise)
            #   AND
            # bias = csc_matrix(np.ones((1, num_documents)))
            # input_data = vstack((input_data, bias)).tocsc()
            # scatter = input_data.dot(input_data.T)

            # We're constucting P and scatter by only iterating through the corpus once.
            ln.debug("building P and scatter matrix.")
            P = np.zeros((self.output_dimensionality, self.input_dimensionality + 1), dtype=float)
            for input_chunk, reduced_chunk in izip(input_data, reduced_representations):
                chunksize = input_chunk.shape[1]
                bias = np.ones((1, chunksize))
                input_chunk = vstack((input_chunk, bias))
                P += reduced_chunk.dot(input_chunk.T)
                scatter += input_chunk.dot(input_chunk.T)


        else:
            ln.debug("Input/Output dim %s" % self.input_dimensionality)
            ln.debug("building scatter matrix from temp files. P will be built in-memory.")
            # In this case, only construct scatter in this strange way, since P is constructed from scatter alone
            for input_chunk in input_data:
                ln.debug("input_chunk shape: (%s,%s)" % input_chunk.shape)
                chunksize = input_chunk.shape[1]

                bias = np.ones((1, chunksize))
                input_chunk = vstack((input_chunk, bias))
                scatter += input_chunk.dot(input_chunk.T)
                del input_chunk

            # construct P in memory
            ln.debug("constructing P")
            P = scatter[:-1, :]
            P *= (1 - self.noise)

        #ln.debug("scatter has shape (%s, %s), P has shape (%s, %s)" % (scatter.shape[0], scatter.shape[1], P.shape[0],P.shape[1]))

        P[:, self.input_dimensionality] *= (1.0 / (1 - self.noise))

        corruption = csc_matrix(np.ones((self.input_dimensionality + 1, 1))) * (1 - self.noise)
        corruption[-1] = 1
        #ln.debug("corruption: %s, %s" % corruption.shape)

        # this is a hacky translation of the original Matlab code, to avoid allocating a big (d+1)x(d+1) matrix
        # instead of element-wise multiplying the matrices, we handle the corresponding areas individually

        # corrupt everything
        Q = scatter * (1-self.noise)**2
        # partially undo corruption to values in (d+1,:)
        Q[self.input_dimensionality] = scatter[self.input_dimensionality] * (1.0/(1-self.noise))
        # partially undo corruption to values in (:,d+1)
        Q[:, self.input_dimensionality] = scatter[:, self.input_dimensionality] * (1.0/(1-self.noise))
        # undo corruption of (-1, -1)
        Q[-1, -1] = scatter[-1, -1] * (1.0/(1-self.noise)**2)

        # replace the diagonal (this is according to the original code again)
        idxs = range(self.input_dimensionality + 1)
        #ln.debug("scatter[idxs,idxs] is %s, %s" % scatter[idxs, idxs].shape)
        #ln.debug("Q[idxs,idxs] is %s, %s" % Q[idxs, idxs].shape)
        #ln.debug("corruption.T is %s, %s" % corruption.T.shape)
        #ln.debug("scatter: %s, Q: %s, corruption:%s" % (scatter.__repr__(), Q.__repr__(), corruption.__repr__()))
        Q[idxs, idxs] = np.squeeze(np.asarray(np.multiply(corruption.todense().T, (scatter[idxs, idxs]))))

        reg = sparse.eye(self.input_dimensionality + 1, format="csc").multiply(self.lambda_)
        reg[-1, -1] = 0

        # W is going to be dx(d+1) (or rx(d+1) for high dimensions)

        # we need to solve W = P * Q^-1
        # Q is symmetric, so Q = Q^T
        # WQ = P
        # (WQ)^T = P^T
        # Q^T W^T = P^T
        # Q W^T = P^T
        # thus, self.weights = np.linalg.lstsq((Q + reg), P.T)[0].T
        ln.debug("solving for weights..")
        # Qreg = (Q + reg) # This is based on Q and reg, and is therefore symmetric
        self.weights = np.linalg.lstsq((Q + reg), P.T)[0].T

        del P
        del Q
        del scatter
        del bias
        del corruption

        ln.info("finished training.")

    def get_hidden_representations(self, input_data):
        # won't work with streams, but also shouldn't need to.
        dimensionality, num_documents = input_data.shape

        bias = csc_matrix(np.ones((1, num_documents)))

        input_data = vstack((input_data, bias)).todense()

        hidden_representations = np.dot(self.weights, input_data)
        if not self.reduce_dimensionality:
            hidden_representations = np.tanh(hidden_representations)

        del input_data
        del bias

        return hidden_representations

    def __getitem__(self, item):
        return self.get_hidden_representations(item)
