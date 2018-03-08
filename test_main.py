import json
import nltk
import math
import re
from pycorenlp import StanfordCoreNLP
from textblob import TextBlob
from nltk.corpus import wordnet
nlp = StanfordCoreNLP('http://localhost:9000')
f = open("sample_sentences.txt","r")
#defining regex for matching POS tags of Noun, Adjective and Verb
pattern_noun = r'^[N]{2}(S|P|PS)?$'
pattern_adj = r'^[J]{2}(R|S)?$'
pattern_verb = r'^VB(D|G|N|P|Z)?$'
pattern_adverb = r'^RB(R|S)?$'

asp_sent = {}
asp_rating = {}
for line in f:
    raw_sentences = TextBlob(line)

def insert_asp_sent(asp,sent):
    if asp not in asp_sent:
        asp_sent[asp] = []
    asp_sent[asp].append(sent)

def getAntonym(word):
    word = str(word)
    max = 999999
    desirable = ""
    polarity = math.fabs(TextBlob(word).sentiment.polarity)
    antonyms = []
    synonyms = []
    definition = []
    syns = wordnet.synsets(word)
    ##Extracting Lemmatized list of antonyms of given word
    for each in syns:
        definition.append(each.definition())
        for l in each.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    print(synonyms)
    print(antonyms)
    print(definition)
    ##Finding desirable word
    for ind_word in antonyms:
        blob_wrapped_word = TextBlob(ind_word)
        ind_rating = blob_wrapped_word.sentiment.polarity
        diff = math.fabs(polarity - math.fabs(ind_rating))
        if(diff < max):
            max = diff
            desirable = ind_word
    return desirable

def existsNegative(adj,dep_output):
    for k in dep_output['sentences'][0]['basicDependencies']:
        gov = k['governorGloss']
        dep = k['dependentGloss']
        if k['dep'] == 'neg' and gov == adj:
            return True
    return False

def testConj(noun,adj):
    for k in dep_output['sentences'][0]['basicDependencies']:
        gov = k['governorGloss']
        dep = k['dependentGloss']
        if k['dep'] == 'conj' and gov == adj:
            insert_asp_sent(noun,dep)
    insert_asp_sent(noun,adj)

def testCompound(noun,adj):
    for k in dep_output['sentences'][0]['basicDependencies']:
        gov = k['governorGloss']
        dep = k['dependentGloss']
        if k['dep'] == 'compound' and gov == noun:
            modified_adj = adj+" "+dep
            insert_asp_sent(noun,modified_adj)

def testAdvModAdj(noun,adj):
    for k in dep_output['sentences'][0]['basicDependencies']:
        gov = k['governorGloss']
        dep = k['dependentGloss']
        if k['dep'] == 'advmod' and gov == adj:
            modified_adj = dep+" "+gov
            insert_asp_sent(noun,modified_adj)

sent_array = raw_sentences.sentences
for ind in sent_array:
    text = str(ind)
    #tokenized = nltk.word_tokenize(text)

    d={}
    pos_output = nlp.annotate(text,properties={
        'annotators': 'pos',
        'outputFormat': 'json'
    })

    dep_output = nlp.annotate(text,properties={
        'annotators': 'depparse',
        'outputFormat': 'json'
    })

    for i in pos_output['sentences'][0]['tokens']:
        d[i['word']] = i['pos']
    for j in dep_output['sentences'][0]['basicDependencies']:
        print(j)
        gov = j['governorGloss']
        dep = j['dependentGloss']
        #for adjectival dependencies
        if j['dep'] == 'amod':
            ##rule 1: adjectival modifier applied to noun subject
            if re.match(pattern_noun,d[gov]) and re.match(pattern_adj,d[dep]):
                ##conj relation between two adjectives
                testConj(gov,dep)
                ##compund relation between nouns
                testCompound(gov,dep)
                ##adjective modified by adverb
                testAdvModAdj(gov,dep)
        #for noun subject dependenices
        elif j['dep'] == 'nsubj':
            ##rule 2: nsubj - adjective link
            if re.match(pattern_adj,d[gov]) and re.match(pattern_noun,d[dep]):
                # check if there exists a negative relationship
                if existsNegative(gov, dep_output):
                    temp = getAntonym(gov)
                    insert_asp_sent(dep,temp)
                else:
                    insert_asp_sent(dep,gov)
            ##rule 3: nusbj-acomp/xcomp (NN<-VBZ->JJ)
            ##rule 4: nsubj-dobj (NN<-VBZ->JJ)
            ##rule 5: nsubj-advmod (NN<-VBZ->JJ)
            elif re.match(pattern_verb,d[gov]) and re.match(pattern_noun,d[dep]):
                for k in dep_output['sentences'][0]['basicDependencies']:
                    temp_gov = k['governorGloss']
                    temp_dep = k['dependentGloss']
                    if k['dep'] == 'xcomp' or k['dep'] == 'acomp':
                        if re.match(pattern_verb,d[temp_gov]) and re.match(pattern_adj,d[temp_dep]) and temp_gov == gov:
                            if existsNegative(temp_gov, dep_output):
                                temp = getAntonym(temp_dep)
                                insert_asp_sent(dep, temp)
                            else:
                                insert_asp_sent(dep, temp_dep)
                    elif k['dep'] == 'dobj':
                        if re.match(pattern_verb,d[temp_gov]) and re.match(pattern_noun,d[temp_dep]) and temp_gov == gov:
                            if existsNegative(temp_gov, dep_output):
                                temp = getAntonym(temp_dep)
                                insert_asp_sent(dep, temp)
                            else:
                                insert_asp_sent(dep, temp_dep)
                    elif k['dep'] == 'advmod':
                        if re.match(pattern_verb,d[temp_gov]) and re.match(pattern_adverb,d[temp_dep]) and temp_gov == gov:
                            if existsNegative(temp_gov, dep_output):
                                temp = getAntonym(temp_dep)
                                insert_asp_sent(dep, temp)
                            else:
                                insert_asp_sent(dep, temp_dep)

print(asp_sent)

#averaging the polarities
for asp in asp_sent:
    length = len(asp_sent[asp])
    avg = 0
    sum = 0
    for word in asp_sent[asp]:
        blob_word = TextBlob(word)
        sum = sum + blob_word.sentiment.polarity
    avg = sum / length
    asp_rating[asp] = avg
print(asp_rating)