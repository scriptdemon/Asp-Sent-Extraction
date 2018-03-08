import json
import nltk
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

text = 'It has lovely camera.'
dep_output = nlp.annotate(text,properties={
    'annotators': 'depparse',
    'outputFormat': 'json'
})
##print(output)
##print(output['sentences'][0])

##tostring = json.dumps(output)
##loaded_json = json.loads(tostring)
##print(tokenized)
print(dep_output['sentences'][0]['basicDependencies'])