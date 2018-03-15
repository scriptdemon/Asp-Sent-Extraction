import math
from textblob import Word
from textblob.wordnet import VERB
from textblob.wordnet import NOUN
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

word_noun_forms = Word("match").get_synsets(pos=VERB)
word_verb_forms = Word("match").get_synsets(pos=NOUN)

word = TextBlob("used")
ant_word = TextBlob("modified")

vad_word = analyser.polarity_scores("Camera is modified.")
vad_word2 = analyser.polarity_scores("Camera is improved.")

print(vad_word)
print(vad_word2)
print(word.sentiment.polarity)
print(ant_word.sentiment.polarity)
#print(word_noun_forms)
#print(word_verb_forms)

