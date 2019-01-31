import os
import gensim

path_to_db = "/media/norpheo/mySQL/db/ssorc"
token_types = ["word",
               #"wordbigram",
               #"pos",
               #"posbigram",
               #"lemma",
               #"lemmabigram",
               #"originalText",
               #"originalTextbigram"
               ]
dic_paths = [os.path.join(path_to_db, "dictionaries", f"pruned_{toty}_lower_notriggers_pd.dict") for toty in token_types]  # input
model_paths = [os.path.join(path_to_db, "models", f"pruned_{toty}_lower_notriggers_pd.tfidf") for toty in token_types]  # output


for i in range(len(token_types)):
    token_type = token_types[i]
    dictionary = gensim.corpora.Dictionary.load(dic_paths[i])
    model_path = model_paths[i]

    print(f"Build TFIDF Model for {token_type}")
    tfidf = gensim.models.TfidfModel(dictionary=dictionary)
    print("Save TFIDF Model.")
    tfidf.save(model_path)


