import os
import mysql.connector
import pickle
import gensim

from src.utils.corpora import AnnotationStream
from src.utils.LoopTimer import LoopTimer
from src.modules.pattern_matching import sentence2tree


def make_deptree(mod_name, dep_tree_dict, dictionary, dep_type='basicDependencies', limit=2000):
    #dep_type = 'enhancedDependencies'
    #dep_type = 'enhancedPlusPlusDependencies'

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="thesis",
    )

    cursor = connection.cursor()
    cursor.execute("USE ssorc;")
    sq1 = f"SELECT abstract_id FROM abstracts_ml WHERE entities LIKE '%machine learning%' AND annotated=1 LIMIT {limit}"
    cursor.execute(sq1)

    print("Collecting Abstracts")
    abstracts = set()
    for row in cursor:
        abstracts.add(row[0])
    connection.close()
    print(f"{len(abstracts)} to build.")

    size = len(abstracts)

    annotations = AnnotationStream(abstracts=abstracts, output='all')

    lc = LoopTimer(update_after=10, avg_length=200, target=size)
    for abstract_id, annotation in annotations:
        for sentence in annotation['sentences']:
            dep_tree = sentence2tree(sentence, dictionary=dictionary, dep_type_=dep_type)
            sentence_id = int(sentence['index'])
            if dep_tree is not None:
                dep_tree_dict[(abstract_id, sentence_id)] = dep_tree
        lc.update("Build Dep Tree Dict")
    print()
    print(f"Size of Dictionary: {len(dictionary)}")


if __name__ == '__main__':
    mod_name = "dep_tree_tiny"
    dtd = dict()
    dtd_dict = gensim.corpora.Dictionary()

    make_deptree(mod_name, dtd, dtd_dict, dep_type='basicDependencies', limit=2000)

    path_to_db = "/media/norpheo/mySQL/db/ssorc"
    dep_tree_path = os.path.join(path_to_db, 'pattern_matching', f"{mod_name}.pickle")
    dict_path = os.path.join(path_to_db, 'pattern_matching', f"{mod_name}.dict")

    with open(dep_tree_path, 'wb') as dt_file:
        pickle.dump(dtd, dt_file, protocol=pickle.HIGHEST_PROTOCOL)
    dtd_dict.save(dict_path)
