import os

import gensim
import src.utils.corpora as corpora


def build_model(dtype, ctype):
    if ctype not in ['words', 'wordbigrams', 'pos', 'posbigrams', 'lemma']:
        print("Type not supported: " + ctype)
        return

    if ctype == 'words':
        dic_type = 'word.dic'
    elif ctype == 'wordbigrams':
        dic_type = 'wordbigram.dic'
    elif ctype == 'pos':
        dic_type = 'pos.dic'
    elif ctype == 'posbigrams':
        dic_type = 'posbigram.dic'
    elif ctype == 'lemma':
        dic_type = 'lemma.dic'

    dirname = os.path.dirname(__file__)
    annotation_dir = os.path.join(dirname, '../../data/processed', dtype, 'annotation')
    tfidf_dir = os.path.join(dirname, '../../data/processed', dtype, 'tfidf')
    dic_dir = os.path.join(dirname, '../../data/processed', dtype, 'dictionaries')

    dictionary = gensim.corpora.Dictionary.load(os.path.join(dic_dir, dic_type))

    file_list = sorted(
        [f for f in os.listdir(annotation_dir) if os.path.isfile(os.path.join(annotation_dir, f))])

    class Corpus(object):
        def __init__(self, ftype, ctype):
            if ctype == 'words':
                self.corpus = corpora.word_doc_stream(ftype, print_status=True)
            elif ctype == 'wordbigrams':
                self.corpus = corpora.wordbigram_doc_stream(ftype, print_status=True)
            elif ctype == 'pos':
                self.corpus = corpora.pos_doc_stream(ftype, print_status=True)
            elif ctype == 'posbigrams':
                self.corpus = corpora.posbigram_doc_stream(ftype, print_status=True)
            elif ctype == 'lemma':
                self.corpus = corpora.lemma_doc_stream(ftype, print_status=True)

        def __iter__(self):
            for entity in self.corpus:
                yield dictionary.doc2bow(entity[1])

    corpus = Corpus(dtype, ctype)
    tfidf = gensim.models.TfidfModel(corpus)

    tfidf.save(os.path.join(tfidf_dir, ctype+'_model.tfidf'))
    print(ctype + ' - Done')
    return tfidf
