mSDA
====

Python implementation of (linear) Marginalized Stacked Denoising Autoencoder (mSDA), as well as dense Cohort of Terms (dCoT). 

Based on Matlab code by Minmin Chen. For original Papers and Code, see http://www.cse.wustl.edu/~mchen/.

Note that the basic mSDA class is pretty much untested at this point, mSDAhd however seems to work okay.

Example usage with dimensional reduction on text:

<pre><code>
from linear_msda import mSDAhd


# load your corpus, should be bag of words format (as in e.g. gensim)

# initialize mSDA / dCoT
msda = mSDAhd(top_k_terms, input_dimensionality=len(dictionary), noise=0.5, num_layers=5)
msda = mSDAhd(dimensions, id2word, noise=noise, num_layers=num_layers)

# train on our corpus, generating the hidden representations
msda.train(preprocessed_bow_documents, chunksize=10000)

# get a hidden representation of new text:
mytext = "............."
bow = preprocess(mytext) # generate bow, remove stopwords, etc.
representation = msda[bow]
# this also works for corpus formats in the same notation, like in gensim

</code></pre>

Use test_msda to run a basic test. What it does is generate a list of pairs of synonyms from the dictionary (using WordNet), and a list or random pairs. It then computes the average similarities of both. The synonyms should of course have a higher result.
This is of course only very primititve test to begin with.
