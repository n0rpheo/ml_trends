import os
import gensim

path_to_db = "/media/norpheo/mySQL/db/ssorc"
token_types = [#"word",
               #"wordbigram",
               #"pos",
               #"posbigram",
               #"lemma",
               #"lemmabigram",
               #"coarse_pos",
               #"coarse_posbigram",
                "merged_word"
               ]
dic_paths = [os.path.join(path_to_db, "dictionaries", f"aiml_full_ner_{toty}.dict") for toty in token_types]  # input
model_paths = [os.path.join(path_to_db, "models", f"aiml_full_ner_{toty}.tfidf") for toty in token_types]  # output


for i in range(len(token_types)):
    token_type = token_types[i]
    dictionary = gensim.corpora.Dictionary.load(dic_paths[i])
    model_path = model_paths[i]

    print(f"Build TFIDF Model for {token_type}")
    tfidf = gensim.models.TfidfModel(dictionary=dictionary)
    print("Save TFIDF Model.")
    tfidf.save(model_path)


