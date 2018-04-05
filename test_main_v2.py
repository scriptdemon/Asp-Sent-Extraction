#import json
#import nltk
#import math
#import re
import string
from pycorenlp import StanfordCoreNLP
from textblob import TextBlob
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import test_main_rules as rules

analyser = SentimentIntensityAnalyzer()
nlp = StanfordCoreNLP('http://localhost:9000')
#f = open("sample_sentences.txt","r")
line = '''
I have received redmi note 4 black matte 64gb version today. Packaging is so good.
About phone: its fabulous phone.Amazing battery back up, good camera, great memory, beautiful colour of phone with classy primium look of black matte makes it different from other phone. I am loving every feature of this phone.
'''

asp_sent = {}
asp_rating = {}

def corefResolver(line):
    ind_sent = []
    complete_coref_output = nlp.annotate(line,properties={'annotators':'dcoref','outputFormat':'json'})
    coref_output = complete_coref_output['corefs']
    raw_sent = TextBlob(line)
    sent_array = raw_sent.sentences
    for j in sent_array:
        ind_sent.append(str(j))
    for k in coref_output:
        prop_noun = ""
        for m in coref_output[k]:
            if (m['type'] == 'NOMINAL' or m['type'] == 'PROPER') and prop_noun == "":
                prop_noun = m['text']
            elif m['type'] == 'PRONOMINAL' and prop_noun != "":
                sent_num = int(m['sentNum'])
                ind_sent[sent_num-1] = ind_sent[sent_num-1].replace(m['text'],prop_noun)

    return ind_sent

#insert aspect-sentiment pair in asp_sent dictionary
def insert_asp_sent(asp,sent):
    if asp not in asp_sent:
        asp_sent[asp] = []
    asp_sent[asp].append(sent)

#get negative relations for further reference
def getNegRelations(dep_output,negatives):
    for j in dep_output['sentences'][0]['basicDependencies']:
        gov = j['governorGloss']
        if j['dep'] == 'neg':
            negatives[gov] = ''
    return negatives
#wrap the sentences in TextBlob and Sentence Tokenize
#for line in f:
    #sent_array = corefResolver(line)
sent_array = corefResolver(line)

count=0
for k in sent_array:
    k = k.lower()
    sent_array[count] = k
    count += 1

for ind in sent_array:
    #print(ind)
    text = str(ind)
    negatives = {}
    d = {}
    rel_dictionary = {}
    pos_output = nlp.annotate(text, properties={
        'annotators': 'pos',
        'outputFormat': 'json'
    })

    dep_output = nlp.annotate(text, properties={
        'annotators': 'depparse',
        'outputFormat': 'json'
    })

    negatives = getNegRelations(dep_output,negatives)

    #making POS tags dictionary
    for i in pos_output['sentences'][0]['tokens']:
        d[i['word']] = i['pos']

    for j in dep_output['sentences'][0]['basicDependencies']:
        dep_name = j['dep']
        gov = j['governorGloss']
        dep = j['dependentGloss']
        if dep_name not in rel_dictionary:
            rel_dictionary[dep_name] = []
        rel_dictionary[dep_name].append({'gov':gov,'dep':dep})
    #print(rel_dictionary)


    #passing through each dependency
    for j in dep_output['sentences'][0]['basicDependencies']:
        #print(j)
        gov = j['governorGloss']
        dep = j['dependentGloss']
        if j['dep'] == 'amod':
            asp_sent = rules.amodRules(gov,dep,d,rel_dictionary,negatives,asp_sent)
        elif j['dep'] == 'nsubj':
            asp_sent = rules.nsubjRules(gov,dep,d,rel_dictionary,negatives,asp_sent)
        elif j['dep'] == 'acl:relcl':
            asp_sent = rules.aclReclRules(gov,dep,d,rel_dictionary,negatives,asp_sent)
        elif j['dep'] == 'dobj':
            sent_intensity = analyser.polarity_scores(ind)
            if not sent_intensity['compound'] == 0:
                asp_sent = rules.dobjRules(gov,dep,d,rel_dictionary,negatives,asp_sent)
        elif j['dep'] == 'nsubjpass':
            sent_intensity = analyser.polarity_scores(ind)
            if not sent_intensity['compound'] == 0:
                asp_sent = rules.nsubjpassRules(gov,dep,d,rel_dictionary,negatives,asp_sent)

print("\nAspect Sentiment Pairs:")
for each in asp_sent:
    string = str(asp_sent[each])
    print(each+" : "+string)

for asp in asp_sent:
    length = len(asp_sent[asp])
    avg = 0
    sum = 0
    for word in asp_sent[asp]:
        #blob_word = TextBlob(word)
        sent_val = analyser.polarity_scores(word)
        sum = sum + sent_val['compound']
    avg = sum / length
    asp_rating[asp] = avg

#scaling on the 0 to 5 scale
for asp in asp_sent:
    non_scaled = asp_rating[asp]
    scaled = (non_scaled + 1)*2.5
    asp_rating[asp] = scaled

print("\nRatings:")
for each in asp_rating:
    string = str(asp_rating[each])
    print(each+" : "+string)