import os
import gensim

path_to_db = "/media/norpheo/mySQL/db/ssorc"
#nlp_model = "en_core_web_sm_nertrained_v3"
nlp_model = "en_core_web_lg_mlalgo_v1"
path_to_dictionaries = os.path.join(path_to_db, "dictionaries", nlp_model)
token_types = ["word",
               "wordbigram",
               #"pos",
               #"posbigram",
               "lemma",
               "lemmabigram",
               #"wordRetokenized",
               #"coarse_pos",
               #"coarse_posbigram",
               "merged_word"
               ]
dic_paths = [os.path.join(path_to_dictionaries, f"full_{toty}.dict") for toty in token_types]

for i in range(len(token_types)):
    token_type = token_types[i]

    dictionary = gensim.corpora.Dictionary.load(dic_paths[i])
    print(f"{token_type}:")
    print(dictionary)
    no_below_num = 0
    while len(dictionary) > 20000:
        no_below_num += 1
        dictionary.filter_extremes(no_below=no_below_num, no_above=0.5, keep_n=None)
    print(f"No Below:{no_below_num}")
    print(f"Len: {len(dictionary)}")
    print("-----------------------")
    new_dic_path = os.path.join(path_to_dictionaries, f"pruned_{token_type}.dict")
    dictionary.save(new_dic_path)
