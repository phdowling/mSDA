mSDA
====

Python implementation of (linear) Marginalized Stacked Denoising Autoencoder (mSDA), as well as dense Cohort of Terms (dCoT). 

Based on Matlab code by Minmin Chen. For original Papers and Code, see http://www.cse.wustl.edu/~mchen/.

Note that the basic mSDA class is pretty much untested at this point, mSDAhd however seems to work okay.

Example usage with dimensional reduction on text:

```python
from linear_msda import mSDAhd

# initialize mSDA / dCoT
msda = mSDAhd(dimensions, id2word, noise=noise, num_layers=num_layers)

# load your corpus, should be bag of words format (as in e.g. gensim)
preprocessed_bow_documents = MmCorpus("test_corpus.mm")

# train on our corpus, generating the hidden representations
msda.train(preprocessed_bow_documents, chunksize=10000)

# get a hidden representation of new text:
mytext = "add some text here"
bow = preprocess(mytext) # remove stopwords, generate bow, etc.
representation = msda[bow]

# this also works for corpus formats in the same notation, like in gensim
mycorpus_raw = ["add some text here", "another text", "this is a document"]
corpus = [preprocess(doc) for doc in mycorpus_raw]
representations = msda[corpus]
```

Use test_msda to run a basic test. What it does is generate a list of pairs of synonyms from the dictionary (using WordNet), and a list or random pairs. It then computes the average similarities of both. The synonyms should have a higher average similarity (in my experiments, around 0.2 compared to 0.1). This is just a preliminary, primitive test to get an idea for whether the generated representations are of any use.
