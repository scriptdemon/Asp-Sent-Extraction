import math
from textblob import Word
from textblob.wordnet import VERB
from textblob.wordnet import NOUN
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

word_noun_forms = Word("match").get_synsets(pos=VERB)
word_verb_forms = Word("match").get_synsets(pos=NOUN)

word = TextBlob("use")
ant_word = TextBlob("I like the camera but hate battery")

vad_word = analyser.polarity_scores("I use this camera")
vad_word2 = analyser.polarity_scores("I like the camera but hate battery")

print(vad_word)
print(vad_word2)
print(word.sentiment.polarity)
print(ant_word.sentiment.polarity)
#print(word_noun_forms)
#print(word_verb_forms)

