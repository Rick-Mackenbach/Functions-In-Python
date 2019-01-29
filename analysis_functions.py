import numpy
import scipy
from scipy.linalg import inv
import sklearn.metrics.pairwise
from sklearn.feature_extraction.text import CountVectorizer


def matrix_median_correlation(x):
    temp = []
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):

            ## Faster computations:
            if (i == j):
                temp.append(1)
                continue
            
            ## Define the vector subsets for calculations ==> faster computations
            xIndex = x[:,i]
            jIndex = x[:,j]

            ## Compute the nominator
            nom1 = (xIndex - numpy.median(xIndex))
            nom2 = (jIndex - numpy.median(jIndex))
            nominTotal = numpy.sum(nom1 * nom2)

            ## Compute the denominator
            denom1 = numpy.sum( (xIndex - numpy.median(xIndex) )**2 )
            denom2 = numpy.sum( (jIndex - numpy.median(jIndex) )**2 )
            denomTotal = (denom1**0.5) * (denom2**0.5)
            
            ## Append the found value to our empty list
            temp.append(nominTotal/denomTotal)
    results = numpy.array(temp)
    return numpy.reshape(results, (x.shape[1],x.shape[1]))
    pass


def r_squared(x,y):
    ## Implementation of the function
    nominator = numpy.sum((y-x)**2)
    denominator = numpy.sum((x - numpy.mean(x))**2)
    return (1 - (nominator/denominator))
    pass


def linear_prediction(x_values,y_target_prediction,x_test):
    
    ## First we define the linear fit which will find the OLS beta values ##
    
    # First we add the intercept, by adding a column of ones to our x_values
    x_values = numpy.c_[x_values, numpy.ones(x_values.shape[0])]
    
    # Calculate the best beta values 
    transposedX = x_values.transpose()
    betweenBrackets = transposedX @ x_values
    inversedXtX = scipy.linalg.inv(betweenBrackets)
    onlyNeedVector = inversedXtX @ transposedX
    beta_values = onlyNeedVector @ y_target_prediction    
    
    
    ## Secondly, we find the prediction. Which will be our beta values * test_data ##
    
    # Add intercept to our x_test dataset
    x_test = numpy.c_[x_test, numpy.ones(x_test.shape[0])]
    
    # Return the prediction
    return x_test @ beta_values



def word_index(x):
    output = {} # dictionary
    index = 0
    for word in x:
        if not word in output:
            output[word] = index
            index += 1
    
    return output

def word_count(x):
    
     # build set of words in a single list
    all_words = []
    for document in x:
        for word in document:
            all_words.append(word)

    # build mapping from words to indexes
    vocab = word_index(all_words)

    #initialise the word_doc_matrix
    word_doc = scipy.sparse.dok_matrix((len(x), len(vocab)))

    #increment entries in word_dox
    doc_number = 0
    for sentence in x:
        for word in sentence:
            word_number = vocab[word]
            word_doc[doc_number, word_number] +=1
        doc_number +=1

    return word_doc


def find_most_similar(sentences):
    a = word_count(sentences)

    ## Compute cosine similarity 
    cosineSimilarity = sklearn.metrics.pairwise.cosine_similarity(a)
    
    ## Change the diagonal entries
    numpy.fill_diagonal(cosineSimilarity,0)
    
    ## Sort
    cosineSimilaritySorted = numpy.argmax(cosineSimilarity, axis=1)

    
    ## Return the most similar sentences
    return (cosineSimilaritySorted)
    pass


## This is an alternative, shorter and cleaner approach ##
def find_most_similar_shorter(sentences):
    vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    
    ## Use CountVectorizer
    count_dataSet = vectorizer.fit_transform(sentences)
    
    ## Compute cosine similarity 
    cosineSimilarity = sklearn.metrics.pairwise.cosine_similarity(count_dataSet)
    
    ## Change diagonal entries
    numpy.fill_diagonal(cosineSimilarity,0)
    
    ## Sort the dataSet
    cosineSimilaritySorted = numpy.argmax(cosineSimilarity, axis=1)
    
    ## Return the most similar sentences
    return (cosineSimilaritySorted)
    pass
