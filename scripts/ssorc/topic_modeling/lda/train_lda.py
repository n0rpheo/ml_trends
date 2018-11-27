import os
import gensim
import scipy.sparse

token_type = 'lemma'
num_topics = 20
update_every = 1
passes = 1
model_name = "tm_lda_lemma.model"

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_models = os.path.join(path_to_db, 'models')
path_to_dictionaries = os.path.join(path_to_db, 'dictionaries')
dic_path = os.path.join(path_to_dictionaries, "full_" + token_type + ".dict")
feature_file_path = os.path.join(path_to_db, 'features', 'tm_lemma_features.npz')

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(dic_path)
print('Load Features')
tm_features = scipy.sparse.load_npz(feature_file_path)

corpus_tfidf = gensim.matutils.Sparse2Corpus(tm_features.transpose())

print('Training Model.')
lda_model = gensim.models.LdaMulticore(corpus=corpus_tfidf,
                                       id2word=dictionary,
                                       num_topics=num_topics,
                                       eval_every=update_every,
                                       passes=passes,
                                       workers=5)


print('Model Trained.')
model_path = os.path.join(path_to_models, model_name)
lda_model.save(model_path)
print('Model Saved.')
