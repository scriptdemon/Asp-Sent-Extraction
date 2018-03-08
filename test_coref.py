from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

text = "The phone beeped and it cracked when I switched the camera on"
coref_output = nlp.annotate(text,properties={
        'annotators': 'dcoref',
        'outputFormat': 'json'
    })

print(coref_output['corefs']['1'])