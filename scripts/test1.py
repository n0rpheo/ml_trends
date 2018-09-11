import json
from stanfordcorenlp import StanfordCoreNLP


def paragraph_splitter(split_abstract):
    max_para_title_words = 3    # Maximum Number of words for a line to be considered as a paragraph_title
    min_para_words = 10         # Minimum Number of words for a line to be considered as a paragraph

    lines = split_abstract.split('\n')
    paragraph_title = ''
    empty_line_allowed = False

    paras = list()

    for line in lines:
        # Check if line matches a Paragraph-Title-Pattern
        if len(line) > 0 and (len(line.split()) <= max_para_title_words or line == line.upper()):
            paragraph_title = line
            empty_line_allowed = True
        elif len(line) > 0 and (len(line.split()) >= min_para_words):
            para_data = dict()
            para_data['paragraphContent'] = line
            para_data['paragraphTitle'] = paragraph_title

            paras.append(para_data)

            empty_line_allowed = False
        elif len(line) == 0 and not empty_line_allowed:
            paragraph_title = ''
    return paras


text = "RESULT:\n\nThis is an example. There is another sentence. It might not work at all.\nCONCLUSION\nLook at your own Face and what you think you are. Have a nice Day."

annotators = 'tokenize,ssplit,lemma'
splitter_annotators = 'ssplit'
nlp = StanfordCoreNLP('../stanford-corenlp-full-2018-02-27')
props = {'annotators': annotators, 'pipelineLanguage': 'en', 'outputFormat': 'json'}
split_props = {'annotators': splitter_annotators, 'pipelineLanguage': 'en', 'outputFormat': 'json'}

paragraphs = paragraph_splitter(text)

last_id = 0
para_info = dict()
para_info['sentences'] = list()
for paragraph in paragraphs:
    para_text = paragraph['paragraphContent']
    para_title = paragraph['paragraphTitle']
    para_anno = json.loads(nlp.annotate(para_text, properties=split_props))

    number_of_sentences = len(para_anno['sentences'])

    for i in range(last_id, last_id+number_of_sentences):
        infos = dict()
        infos['label'] = para_title
        infos['index'] = i

        para_info['sentences'].append(infos)

    last_id += number_of_sentences


para_info_text = json.dumps(para_info)

new_text = " ".join([x['paragraphContent'] for x in paragraphs])

annotation = nlp.annotate(new_text, properties=props)

anno = json.loads(annotation)

nlp.close()
