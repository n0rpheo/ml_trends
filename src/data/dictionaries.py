import os
import gensim

import src.utils.corpora as corpora
from src.utils.LoopTimer import LoopTimer


def make_dictionaries(dtype):
    dirname = os.path.dirname(__file__)
    inter_dir = os.path.join(dirname, '../../data/interim', dtype)

    word_dic = gensim.corpora.Dictionary()
    pos_dic = gensim.corpora.Dictionary()
    lemma_dic = gensim.corpora.Dictionary()
    wordbi_dic = gensim.corpora.Dictionary()
    posbi_dic = gensim.corpora.Dictionary()

    word_corpus = corpora.word_doc_stream(dtype)
    wordbigram_corpus = corpora.wordbigram_doc_stream(dtype)
    pos_corpus = corpora.pos_doc_stream(dtype)
    posbigram_corpus = corpora.posbigram_doc_stream(dtype)
    lemma_corpus = corpora.lemma_doc_stream(dtype)

    lt = LoopTimer()
    for word_doc, wordbigram_doc, pos_doc, posbigram_doc, lemma_doc in zip(word_corpus, wordbigram_corpus, pos_corpus, posbigram_corpus, lemma_corpus):

        lemma_dic.add_documents([lemma_doc[1]], prune_at=20000000)
        word_dic.add_documents([word_doc[1]], prune_at=20000000)
        wordbi_dic.add_documents([wordbigram_doc[1]], prune_at=20000000)
        pos_dic.add_documents([pos_doc[1]], prune_at=20000000)
        posbi_dic.add_documents([posbigram_doc[1]], prune_at=20000000)

        lt.update("Build Dictionaries")

    lemma_dic.save(os.path.join(inter_dir, 'full_lemma.dict'))
    wordbi_dic.save(os.path.join(inter_dir, 'full_wordbi.dict'))
    word_dic.save(os.path.join(inter_dir, 'full_word.dict'))
    posbi_dic.save(os.path.join(inter_dir, 'full_posbi.dict'))
    pos_dic.save(os.path.join(inter_dir, 'full_pos.dict'))

    print(word_dic)
    print(wordbi_dic)
    print(pos_dic)
    print(posbi_dic)
    print(lemma_dic)
