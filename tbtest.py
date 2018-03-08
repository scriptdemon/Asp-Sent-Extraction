import math
from textblob import Word
from textblob.wordnet import VERB
from textblob.wordnet import NOUN
from textblob import TextBlob

word_noun_forms = Word("match").get_synsets(pos=VERB)
word_verb_forms = Word("match").get_synsets(pos=NOUN)

word = TextBlob("good")
ant_word = TextBlob("amazingly good")


print(word.sentiment.polarity)
print(ant_word.sentiment.polarity)
#print(word_noun_forms)
#print(word_verb_forms)

