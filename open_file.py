import json
import gzip
from pycorenlp import StanfordCoreNLP
# initialize connection with CoreNLP Server
nlp = StanfordCoreNLP('http://localhost:9000')
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))


f = open("output.strict", 'w')
i=0
for l in parse("reviews_Books_5.json.gz"):
    if i>0:
        break
    loaded_json = json.loads(l)
    text = loaded_json["reviewText"]
    output = nlp.annotate(text, properties={
        'annotators': 'pos',
        'outputFormat': 'json'
    })
    print(output)
    i=i+1


