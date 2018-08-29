import gensim

import src.data.make_data as md
import src.data.dictionaries as dictionaries
import src.features.make_features as mf
import src.models.tfidf as tfidf

# Turn Raw Data into structured json Files
#md.make_ssorc_data()

# Split die Abstract into Paragraphs
#md.paragraph_splitter(dtype='ssorc')

# Annotate the Paragraphs with stanfordCoreNLP
#mf.nlp_annotation(dtype='ssorc', annotators="tokenize,ssplit,lemma,pos,depparse,ner")


# Make Dictionaries
#dictionaries.make_dictionaries("ssorc")

""""
# Filter the Dictionaries and save them in the Dictionaries Folder
word_dic = gensim.corpora.Dictionary.load("../data/interim/ssorc/full_word.dict")
wordbi_dic = gensim.corpora.Dictionary.load("../data/interim/ssorc/full_wordbi.dict")
lemma_dic = gensim.corpora.Dictionary.load("../data/interim/ssorc/full_lemma.dict")
pos_dic = gensim.corpora.Dictionary.load("../data/interim/ssorc/full_pos.dict")
posbi_dic = gensim.corpora.Dictionary.load("../data/interim/ssorc/full_posbi.dict")

print("Pruning Word Dictionary")
word_dic.filter_extremes(no_below=5, no_above=0.2, keep_n=10000)

print("Pruning Word Bigram Dictionary")
wordbi_dic.filter_extremes(no_below=5, no_above=0.2, keep_n=30000)

print("Pruning Lemma Dictionary")
lemma_dic.filter_extremes(no_below=5, no_above=0.2, keep_n=10000)

print(word_dic)
print(wordbi_dic)
print(lemma_dic)

word_dic.save('../data/processed/ssorc/dictionaries/word.dic')
wordbi_dic.save('../data/processed/ssorc/dictionaries/wordbigram.dic')
pos_dic.save('../data/processed/ssorc/dictionaries/pos.dic')
posbi_dic.save('../data/processed/ssorc/dictionaries/posbigram.dic')
lemma_dic.save('../data/processed/ssorc/dictionaries/lemma.dic')


# Build the tdidf-Models
word_tfidf = tfidf.build_model('ssorc', ctype='words')
wordbi_tfidf = tfidf.build_model('ssorc', ctype='wordbigrams')
pos_tfidf = tfidf.build_model('ssorc', ctype='pos')
posbi_tfidf = tfidf.build_model('ssorc', ctype='posbigrams')
lemma_tfidf = tfidf.build_model('ssorc', ctype='lemma')
"""

# RF Labeling
# md.rf_label_ssorc_paragraphs()