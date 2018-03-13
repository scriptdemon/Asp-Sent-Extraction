import json
import nltk
import math
import re
import string
from pycorenlp import StanfordCoreNLP
from textblob import TextBlob
from nltk.corpus import wordnet
import test_main_rules as rules

nlp = StanfordCoreNLP('http://localhost:9000')
f = open("sample_sentences.txt","r")

asp_sent = {}
asp_rating = {}

def corefResolver(line):
    ind_sent = []
    complete_coref_output = nlp.annotate(line,properties={'annotators':'dcoref','outputFormat':'json'})
    coref_output = complete_coref_output['corefs']
    print(coref_output)
    raw_sent = TextBlob(line)
    sent_array = raw_sent.sentences
    for j in sent_array:
        ind_sent.append(str(j))
    for k in coref_output:
        prop_noun = ""
        for m in coref_output[k]:
            if m['type'] == 'NOMINAL' and prop_noun == "":
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

#wrap the sentences in TextBlob and Sentence Tokenize
for line in f:
    ##print(type(line))
    ##raw_sentences = TextBlob(line)
    sent_array = corefResolver(line)

for each in sent_array:
    print(each)

for ind in sent_array:
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
    print(rel_dictionary)


    #passing through each dependency
    for j in dep_output['sentences'][0]['basicDependencies']:
        gov = j['governorGloss']
        dep = j['dependentGloss']
        if j['dep'] == 'amod':
            asp_sent = rules.amodRules(gov,dep,d,rel_dictionary,negatives,asp_sent)
        if j['dep'] == 'nsubj':
            asp_sent = rules.nsubjRules(gov,dep,d,rel_dictionary,negatives,asp_sent)