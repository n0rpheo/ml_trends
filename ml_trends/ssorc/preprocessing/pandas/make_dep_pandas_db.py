import os
import mysql.connector
import pandas as pd
import operator
import pickle

import src.modules.pattern_matching as pm
from src.utils.corpora import AnnotationStream
from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
db_path = "/media/norpheo/mySQL/db/ssorc/pandas"

dep_type = 'basicDependencies'
#dep_type = 'enhancedDependencies'
#dep_type = 'enhancedPlusPlusDependencies'

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="thesis",
)

cursor = connection.cursor()
cursor.execute("USE ssorc;")
sq1 = "SELECT abstract_id FROM abstracts_ml WHERE entities LIKE '%machine learning%' AND annotated=1"
cursor.execute(sq1)

print("Collecting Abstracts")
abstracts = set()
for row in cursor:
    abstracts.add(row[0])
connection.close()

annotations = AnnotationStream(abstracts=abstracts, output='all')

lc = LoopTimer(update_after=100, avg_length=200, target=len(abstracts))

abstract_id_list = list()
sentence_id_list = list()
dep_list = list()

for abstract_id, annotation in annotations:
    for sentence in annotation['sentences']:
        sent_id = sentence['index']
        dep_tree = pm.sentence2tree(sentence, dep_type_=dep_type)
        if dep_tree is not None:
            abstract_id_list.append(abstract_id)
            sentence_id_list.append(sent_id)
            dep_list.append(dep_tree)
    num = lc.update("Predict")

dep_df = pd.DataFrame([[abstract_id_list[i], sentence_id_list[i], dep_list[i]] for i in range(len(dep_list))],
                      columns=['abstract_id', 'sentence_id', 'dependency_graph'])
dep_df.to_pickle(os.path.join(db_path, 'ml_dependency_graphs.pandas'))