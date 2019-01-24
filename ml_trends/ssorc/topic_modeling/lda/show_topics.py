import os
import gensim

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_models = os.path.join(path_to_db, 'models')
path_to_dictionaries = os.path.join(path_to_db, 'dictionaries')

model_name = 'tm_lda_lemma.model'
dictionary_name = 'full_lemma.dict'
path_to_model = os.path.join(path_to_models, model_name)
dictionary_path = os.path.join(path_to_dictionaries, dictionary_name)

print('Load Dict')
dictionary = gensim.corpora.Dictionary.load(dictionary_path)
print('Load Model')
lda_model = gensim.models.LdaMulticore.load(path_to_model)

for tid in range(0, lda_model.num_topics):
    topic_terms = lda_model.get_topic_terms(tid, topn=10)

    print(f"Topic {tid}:")

    topics = list()
    for topic_term in topic_terms:
        term_id = topic_term[0]
        topics.append(dictionary[term_id])
    print(", ".join(topics))