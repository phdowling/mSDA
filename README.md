mSDA
====

Python implementation of (linear) Marginalized Stacked Denoising Autoencoder (mSDA), as well as dense Cohort of Terms (dCoT), which is a dimensionality-reduction algorithm based on mSDA.

Based on Matlab code by Minmin Chen. For original Papers and Code, see http://www.cse.wustl.edu/~mchen/.

This code has not been extensively tested, so do not rely on this to in fact produce correct representations quite yet. Keep following this repository to keep up to date.

Example usage with dimensional reduction on text:

```python
from linear_msda import mSDA

# load your corpus, should be bag of words format (as in e.g. gensim)
preprocessed_bow_documents = MmCorpus("test_corpus.mm")

# load your dictionary
id2word = Dictionary("...")

dimensions = 1000

# select prototype word IDs, e.g. by finding the most frequent terms
prototype_ids = [id_ for id_, freq in sorted(id2word.dfs.items(), key=lambda (k, v): v, reverse=True)[:dimensions]]

# initialize mSDA / dCoT
msda = mSDA(noise=0.5, num_layers=3, input_dimensionality=len(id2word), output_dimensionality=dimensions, prototype_ids=prototype_ids)

# train on our corpus, generating the hidden representations
msda.train(preprocessed_bow_documents, chunksize=10000)

# get a hidden representation of new text: (note: this is slow)
mytext = "add some text here"
bow = preprocess(mytext) # remove stopwords, generate bow, etc.
representation = msda[bow]

# this also works for corpus formats in the same notation, like in gensim (this way is also more efficient)
mycorpus_raw = ["add some text here", "another text", "this is a document"]
corpus = [preprocess(doc) for doc in mycorpus_raw]
representations = msda[corpus]
```

Note that this implementation is significantly more efficient when documents are transformed in bulk. If you transform documents one at a time, you may experience runtimes that are orders of magniture slower than those of bulk-processing the same data.
