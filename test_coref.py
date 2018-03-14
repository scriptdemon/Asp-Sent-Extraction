from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

text = "Sam said he would do it."
coref_output = nlp.annotate(text,properties={
        'annotators': 'dcoref',
        'outputFormat': 'json'
    })
#awesome
print(coref_output['corefs'])