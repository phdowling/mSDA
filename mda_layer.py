__author__ = 'dowling'

import numpy as np
import logging
ln = logging.getLogger("mDA")
ln.setLevel(logging.DEBUG)

from scipy import sparse
from scipy.sparse import vstack, csc_matrix, csr_matrix
from gensim import utils, matutils

USE_MAPREDUCE = False


def todense(matrix):
    try:
        return matrix.todense()
    except AttributeError:
        return matrix

class FilteringDualGrouper(object):
    """
    Wrapper for simultaneously iterating over a corpus and its projection into a subset of dimensions.
    """
    def __init__(self, corpus, num_terms, filter_dimensions=None, chunksize=10000, dense=False):
        self.corpus = corpus
        self.num_terms = num_terms
        self.filter_dimensions = filter_dimensions
        self.chunksize = chunksize
        self.dense = dense

    def __iter__(self):
        for chunk_no, chunk in enumerate(utils.grouper(self.corpus, self.chunksize)):
            nnz = sum(len(doc) for doc in chunk)
            # construct the job as a sparse matrix, to minimize memory overhead
            # definitely avoid materializing it as a dense matrix!
            # ln.debug("converting corpus to csc format")
            if self.dense:
                job = matutils.corpus2dense(chunk, num_docs=len(chunk), num_terms=self.num_terms)
            else:
                job = matutils.corpus2csc(chunk, num_docs=len(chunk), num_terms=self.num_terms, num_nnz=nnz)

            if self.filter_dimensions is not None:
                filtered = job[self.filter_dimensions, :]
            else:
                filtered = None

            yield job, filtered
            del chunk


class mDALayer(object):
    def __init__(self, noise, lambda_, input_dimensionality, output_dimensionality=None, prototype_ids=None):
        self.noise = noise
        self.lambda_ = lambda_
        self.input_dimensionality = input_dimensionality

        self.output_dimensionality = output_dimensionality or input_dimensionality
        if self.output_dimensionality != self.input_dimensionality:
            if prototype_ids is None:
                ln.warn("Need prototype IDs to train reduction layer.")

        self.randomized_indices = list(utils.grouper(np.random.permutation(self.input_dimensionality),
                                                     self.output_dimensionality))
        for idx_batch in self.randomized_indices:
            idx_batch.sort()  # should be more efficient when selecting array rows in order later on

        self.prototype_ids = prototype_ids

        self.num_folds = int(np.ceil(float(self.input_dimensionality) / self.output_dimensionality))
        self.blocks = []

    def train(self, corpus, numpy_chunk_input=False, chunksize=10000):
        if self.input_dimensionality != self.output_dimensionality:
            assert self.prototype_ids is not None, "Need prototype IDs to train dimensional reduction layer."

        if self.input_dimensionality != self.output_dimensionality:
            ln.info("mDA reduction layer with %s input and %s output dimensions is beginning training..",
                    self.input_dimensionality, self.output_dimensionality)
            ln.debug("Training the initial dimensional reduction with %s folds" % self.num_folds)
        else:
            ln.info("Training mDA layer with %s dimensions.", self.input_dimensionality)

        # build scatter matrices
        ln.info("Building all scatter and P matrices (full corpus iteration).")
        if numpy_chunk_input:
            ln.debug("Assuming that corpus is made up of numpy chunks.")
            dualIterator = ((chunk, None) for chunk in corpus)
        else:
            dualIterator = FilteringDualGrouper(corpus, self.input_dimensionality, self.prototype_ids, chunksize)

        scatters_and_P_matrices_dict = dict()
        processed = 0
        for chunk_no, (doc_chunk, target_representation_chunk) in enumerate(dualIterator):

            for dim_batch_idx in range(self.num_folds):
                block_indices = self.randomized_indices[dim_batch_idx]
                block_data = doc_chunk[block_indices]

                blocksize = block_data.shape[1]
                bias = np.ones((1, blocksize))
                if type(doc_chunk) == sparse.csc.csc_matrix:
                    input_chunk = vstack((block_data, bias))
                else:
                    input_chunk = np.vstack((block_data, bias))

                scatter_update = todense(input_chunk.dot(input_chunk.T))

                # we only explicitly construct P when we do dimensional reduction, otherwise we can use scatter
                if target_representation_chunk is not None:
                    P_update = todense(target_representation_chunk.dot(input_chunk.T))

                if dim_batch_idx not in scatters_and_P_matrices_dict:
                    scatter = scatter_update

                    # we only explicitly construct P when we do dimensional reduction, otherwise we can use scatter
                    if target_representation_chunk is not None:
                        P = P_update
                    else:
                        P = None
                else:
                    scatter, P = scatters_and_P_matrices_dict[dim_batch_idx]
                    scatter += scatter_update
                    if P is not None:
                        P += P_update

                scatters_and_P_matrices_dict[dim_batch_idx] = (scatter, P)

            processed += blocksize

            ln.info("Processed %s chunks (%s documents)", chunk_no + 1, processed)

        scatters_and_P_matrices = [(scatter, P) for dim_batch_idx, (scatter, P) in
                                   sorted(scatters_and_P_matrices_dict.items())]

        if self.input_dimensionality != self.output_dimensionality:
            ln.info("Computing all reduction layer weights.")
        else:
            ln.info("Computing mDA layer weights.")

        for block_num, (scatter_matrix, P) in enumerate(scatters_and_P_matrices):
            if P is None:
                # we use scatter to compute P
                P = scatter_matrix[:-1, :].copy()
                #P = scatter_matrix.copy()
            P[:, :-1] *= (1 - self.noise)  # apply noise (except last column)

            # P[:, self.input_dimensionality] *= (1.0 / (1 - self.noise))  # undo noise for bias column

            weights = self._computeWeights(scatter_matrix, P)
            if block_num % 10 == 9:
                ln.debug("layer trained up to fold %s/%s..", block_num + 1, self.num_folds)

            self.blocks.append(weights)
        if self.input_dimensionality != self.output_dimensionality:
            ln.info("mDA reduction layer completed training.")
        else:
            ln.info("mDA layer completed training.")

    def _computeWeights(self, scatter, P):
        block_input_d = scatter.shape[0] - 1
        assert scatter.shape[0] == scatter.shape[1]
        assert P.shape == (self.output_dimensionality, block_input_d + 1)

        # DIMENSIONS OVERVIEW
        # r: overall output dimensionality
        # d: input dimensionality of this block

        # scatter: always (d+1) x (d+1)
        # P: in normal mDA d x (d+1), else r x (d+1)
        # Q: same as scatter
        # W: in normal mDA d x (d+1), else r x (d+1)

        # we do the following in limited memory with a streamed corpus to more documents

        #ln.debug("Block input dim: %s. Output dim: %s" % (self.input_dimensionality, self.output_dimensionality))

        corruption = csc_matrix(np.ones((block_input_d + 1, 1))) * (1 - self.noise)
        corruption[-1] = 1
        #ln.debug("corruption: %s, %s" % corruption.shape)

        # this is a hacky translation of the original Matlab code, to avoid allocating a big (d+1)x(d+1) matrix
        # instead of element-wise multiplying the matrices, we handle the corresponding areas individually

        # corrupt everything
        Q = scatter * (1-self.noise)**2
        # partially undo corruption to values in (d+1,:)
        Q[block_input_d] = scatter[block_input_d] * (1.0/(1-self.noise))
        # partially undo corruption to values in (:,d+1)
        Q[:, block_input_d] = scatter[:, block_input_d] * (1.0/(1-self.noise))
        # undo corruption of (-1, -1)
        Q[-1, -1] = scatter[-1, -1] * (1.0/(1-self.noise)**2)

        # replace the diagonal (this is according to the original code again)
        idxs = range(block_input_d + 1)

        Q[idxs, idxs] = np.squeeze(np.asarray(np.multiply(corruption.todense().T, (scatter[idxs, idxs]))))

        reg = sparse.eye(block_input_d + 1, format="csc").multiply(self.lambda_)

        reg[-1, -1] = 0

        # W is going to be dx(d+1) (or rx(d+1) for high dimensions)

        # we need to solve W = P * Q^-1
        # Q is symmetric, so Q = Q^T
        # WQ = P
        # (WQ)^T = P^T
        # Q^T W^T = P^T
        # Q W^T = P^T
        # thus, self.weights = np.linalg.lstsq((Q + reg), P.T)[0].T
        #ln.debug("solving for weights...")
        # Qreg = (Q + reg) # This is based on Q and reg, and is therefore symmetric
        weights = np.linalg.lstsq((Q + reg), P.T)[0].T
        #ln.debug("weights matrix shape is %s x %s. Input dim is %s, output %s. r_dim is %s." %
        #         (weights.shape[0],
        #          weights.shape[1],
        #          self.input_dimensionality,
        #          self.output_dimensionality,
        #          block_input_d))

        del P
        del Q
        del scatter
        del corruption

        return weights

    @staticmethod
    def _get_intermediate_representations(block_weights, block_input_data):
        # both matrices should always be dense at this point

        dimensionality, num_documents = block_input_data.shape

        bias = np.ones((1, num_documents))

        block_input_data = np.vstack((block_input_data, bias))

        hidden_representations = np.dot(block_weights, block_input_data)

        del block_input_data
        del bias

        return hidden_representations

    def _get_hidden_representations(self, input_data):
        # input_data is sparse on the first layer, but it shouldn't matter here
        assert input_data.shape[0] == self.input_dimensionality
        representation_avg = None  # todo don't build incremental avg, just do it the normal way? Might overflow though
        for dim_batch_idx in range(self.num_folds):
            block_indices = self.randomized_indices[dim_batch_idx]
            block_data = input_data[block_indices]

            try:
                block_data = block_data.todense()
            except AttributeError:
                pass

            block_weights = self.blocks[dim_batch_idx]

            block_hidden = self._get_intermediate_representations(block_weights, block_data)

            if representation_avg is None:
                representation_avg = block_hidden
            else:
                representation_avg += (1.0 / (dim_batch_idx + 1)) * (block_hidden - representation_avg)
        representation_avg = np.tanh(representation_avg)

        return representation_avg

    def __getitem__(self, input_data):
        # accepts only numpy/scipy matrices (dense or sparse), and returns a dense matrix in every case

        return self._get_hidden_representations(input_data)

