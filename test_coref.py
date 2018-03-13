from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

text = "Redmi note 5 is a great device. It has high capacity battery. Samsung Galaxy J7 is a boring phone. "
coref_output = nlp.annotate(text,properties={
        'annotators': 'dcoref',
        'outputFormat': 'json'
    })
#awesome
print(coref_output['corefs'])