from nltk.corpus import wordnet
from textblob import TextBlob
import math

def getAntonym(word):
    word = str(word)
    max = 999999
    desirable = ""
    polarity = math.fabs(TextBlob(word).sentiment.polarity)
    antonyms = []
    syns = wordnet.synsets(word)
    ##Extracting Lemmatized list of antonyms of given word
    for each in syns:
        for l in each.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    ##Finding desirable word
    for ind_word in antonyms:
        blob_wrapped_word = TextBlob(ind_word)
        ind_rating = blob_wrapped_word.sentiment.polarity
        diff = math.fabs(polarity - math.fabs(ind_rating))
        if(diff < max):
            max = diff
            desirable = ind_word
    return desirable

getAntonym("good")