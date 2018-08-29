import os
import gensim


dtype = "ssorc"

dirname = os.path.dirname(__file__)
inter_dir = os.path.join(dirname, '../../data/interim', dtype)
dic_dir = os.path.join(dirname, '../../data/processed', dtype, 'dictionaries')

word_dic = gensim.corpora.Dictionary.load(os.path.join(inter_dir, 'full_word.dict'))
wordbi_dic = gensim.corpora.Dictionary.load(os.path.join(inter_dir, 'full_wordbi.dict'))
lemma_dic = gensim.corpora.Dictionary.load(os.path.join(inter_dir, 'full_lemma.dict'))
pos_dic = gensim.corpora.Dictionary.load(os.path.join(inter_dir, 'full_pos.dict'))
posbi_dic = gensim.corpora.Dictionary.load(os.path.join(inter_dir, 'full_posbi.dict'))

print("Pruning Word Dictionary")
word_dic.filter_extremes(no_below=5, no_above=0.2, keep_n=10000)
# word_dic.filter_extremes(no_below=3, no_above=0.4, keep_n=100000000)

print("Pruning Word Bigram Dictionary")
wordbi_dic.filter_extremes(no_below=5, no_above=0.2, keep_n=30000)
# wordbi_dic.filter_extremes(no_below=3, no_above=0.4, keep_n=100000000)

print("Pruning Lemma Dictionary")
lemma_dic.filter_extremes(no_below=5, no_above=0.2, keep_n=10000)
# lemma_dic.filter_extremes(no_below=3, no_above=0.4, keep_n=100000000)

print(word_dic)
print(wordbi_dic)
print(lemma_dic)

word_dic.save(os.path.join(dic_dir, 'word.dic'))
wordbi_dic.save(os.path.join(dic_dir, 'wordbigram.dic'))
pos_dic.save(os.path.join(dic_dir, 'pos.dic'))
posbi_dic.save(os.path.join(dic_dir, 'posbigram.dic'))
lemma_dic.save(os.path.join(dic_dir, 'lemma.dic'))