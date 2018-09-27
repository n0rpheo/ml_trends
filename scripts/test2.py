import os
import gensim


feature_file_name = 'tm_features'
token_type = 'word'
num_samples = 1000

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dictionaries = os.path.join(path_to_db, "dictionaries")
path_to_models = os.path.join(path_to_db, 'models')
path_to_feature_file = os.path.join(path_to_db, 'features', feature_file_name + '.npz')
dic_path = os.path.join(path_to_dictionaries, "full_" + token_type + ".dict")
tfidf_path = os.path.join(path_to_models, token_type + "_model.tfidf")

print('Load Dictionary')
dictionary = gensim.corpora.Dictionary.load(dic_path)
print('Load TFIDF')
tfidf = gensim.models.TfidfModel.load(tfidf_path)

document1 = ['This', 'is', 'some', 'text', '.', '.', '.']
document2 = ['This', 'is', 'another', 'sentence', '.']

documents = [document1, document2]
bow = dictionary.doc2bow(document1)
tfidf_vec = tfidf[bow]
print(tfidf_vec)
