import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """Returns the preprocess list of words from a twitter text corpus.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove symbols representing currency, stock market ticker, etc. like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # remove hash sign from hashtags
    tweet = re.sub(r'#', '', tweet)

    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        # remove stopwords and punctuations
        if (word not in stopwords_english and word not in string.punctuation):
            # stemming word
            stem_word = stemmer.stem(word)  
            tweets_clean.append(stem_word)

    return tweets_clean
    

def build_freqs(tweets, labels):
    """Returns a dictionary containing words as keys and their respective
    counts as values.
    Input:
        tweets: a list of tweets
        labels: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # converting np array to list since zip needs an iterable
    # the squeeze is necessary or else the list will end up with one element
    labels_list = np.squeeze(labels).tolist()

    # starting with an empty dictionary and populating it by looping over all tweets
    # and over all processed words in each tweet
    freqs = {}
    for label, tweet in zip(labels_list, tweets):
        for word in process_tweet(tweet):
            pair = (word, label)
            freqs[pair] = freqs.get(pair, 0) + 1

    return freqs