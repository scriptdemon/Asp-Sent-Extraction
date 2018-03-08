from pycorenlp import StanfordCoreNLP
from nltk.corpus import wordnet
nlp = StanfordCoreNLP('http://localhost:9000')
text = 'Camera is awesome'
syns = wordnet.synsets("beautiful")

print(syns)

##get antonyms
antonyms = []
for each in syns:
    for l in each.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())