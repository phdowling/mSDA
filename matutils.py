__author__ = 'dowling'
import numpy
import scipy
from six import itervalues

import logging
logger = logging.getLogger("msda.matutils")

# These are some utility functions, taken from Gensim by Radim Rekurek. They're included here for portability only.
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

def sparse2full(doc, length):
    """
    Convert a document in sparse document format (=sequence of 2-tuples) into a dense
    numpy array (of size `length`).

    This is the mirror function to `full2sparse`.

    """
    result = numpy.zeros(length, dtype=numpy.float32)  # fill with zeroes (default value)
    doc = dict(doc)
    # overwrite some of the zeroes with explicit values
    result[list(doc)] = list(itervalues(doc))
    return result

def full2sparse(vec, eps=1e-9):
    """
    Convert a dense numpy array into the sparse document format (sequence of 2-tuples).

    Values of magnitude < `eps` are treated as zero (ignored).

    This is the mirror function to `sparse2full`.

    """
    vec = numpy.asarray(vec, dtype=float)
    nnz = numpy.nonzero(abs(vec) > eps)[0]
    return list(zip(nnz, vec.take(nnz)))


def corpus2csc(corpus, num_terms=None, dtype=numpy.float64, num_docs=None, num_nnz=None, printprogress=0):
    """
    Convert a streamed corpus into a sparse matrix, in scipy.sparse.csc_matrix format,
    with documents as columns.

    If the number of terms, documents and non-zero elements is known, you can pass
    them here as parameters and a more memory efficient code path will be taken.

    The input corpus may be a non-repeatable stream (generator).

    This is the mirror function to `Sparse2Corpus`.

    """
    try:
        # if the input corpus has the `num_nnz`, `num_docs` and `num_terms` attributes
        # (as is the case with MmCorpus for example), we can use a more efficient code path
        if num_terms is None:
            num_terms = corpus.num_terms
        if num_docs is None:
            num_docs = corpus.num_docs
        if num_nnz is None:
            num_nnz = corpus.num_nnz
    except AttributeError:
        pass # not a MmCorpus...
    if printprogress:
        logger.info("creating sparse matrix from corpus")
    if num_terms is not None and num_docs is not None and num_nnz is not None:
        # faster and much more memory-friendly version of creating the sparse csc
        posnow, indptr = 0, [0]
        indices = numpy.empty((num_nnz,), dtype=numpy.int32) # HACK assume feature ids fit in 32bit integer
        data = numpy.empty((num_nnz,), dtype=dtype)
        for docno, doc in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info("PROGRESS: at document #%i/%i" % (docno, num_docs))
            posnext = posnow + len(doc)
            indices[posnow: posnext] = [feature_id for feature_id, _ in doc]
            data[posnow: posnext] = [feature_weight for _, feature_weight in doc]
            indptr.append(posnext)
            posnow = posnext
        assert posnow == num_nnz, "mismatch between supplied and computed number of non-zeros"
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    else:
        # slower version; determine the sparse matrix parameters during iteration
        num_nnz, data, indices, indptr = 0, [], [], [0]
        for docno, doc in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info("PROGRESS: at document #%i" % (docno))
            indices.extend([feature_id for feature_id, _ in doc])
            data.extend([feature_weight for _, feature_weight in doc])
            num_nnz += len(doc)
            indptr.append(num_nnz)
        if num_terms is None:
            num_terms = max(indices) + 1 if indices else 0
        num_docs = len(indptr) - 1
        # now num_docs, num_terms and num_nnz contain the correct values
        data = numpy.asarray(data, dtype=dtype)
        indices = numpy.asarray(indices)
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    return result

