import json

from stanfordcorenlp import StanfordCoreNLP


annotators = 'tokenize,ssplit,pos,lemma,depparse'
nlp = StanfordCoreNLP('../stanford-corenlp-full-2018-02-27')
props = {'annotators': annotators, 'pipelineLanguage': 'en', 'outputFormat': 'json'}


text = "..."
anno = nlp.annotate(text, properties=props)
para_anno = json.loads(anno)

print(para_anno)

nlp.close()