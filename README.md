mSDA
====

Python implementation of (linear) Marginalized Stacked Denoising Autoencoder (mSDA), as well as dense Cohort of Terms (dCoT). 

Based on Matlab code by Minmin Chen. For original Papers and Code, see http://www.cse.wustl.edu/~mchen/.

Example usage with dimensional reduction on text:

<code>
from linear_msda import mSDA

# stuff you need to do yourself:
# - load your corpus, should be bag of words format (as in e.g. gensim)
# - select the top k most frequent terms (or whatever you want your prototype terms to be based on)

# initialize mSDA / dCoT
msda = mSDAhd(top_k_terms, len(dictionary), noise=0.5, num_layers=5)

# train on our corpus, generating the hidden representations
representations = msda.train(preprocessed_bow_documents, return_hidden=True)

# compute the similarity of two documents
similarity = cosine_similarity(representations[0], representations[1])

# get a hidden representation of new text:
mytext = "............."
bow = preprocess(mytext) # generate bow, remove stopwords, etc.
representation = msda.get_hidden_representations([bow])
</code>

The code is still in it's early stages and probably has a good amount of bugs at this point.
