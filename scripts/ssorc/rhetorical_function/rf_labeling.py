from src.modules.RhetoricalFunctionLabeling import PatternMatching
import mysql.connector


dep_type = 'basicDependencies'
#dep_type = 'enhancedDependencies'
#dep_type = 'enhancedPlusPlusDependencies'

load_from_file = True


pm = PatternMatching(dep_type)

if load_from_file:
    pm.load_dep_tree("dep_tree_dict.pickle")
else:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="thesis",
    )

    cursor = connection.cursor()
    cursor.execute("USE ssorc;")
    # sq1 = f"SELECT abstract_id FROM abstracts WHERE annotated=1 and dictionaried=1 LIMIT 10000"
    sq1 = f"SELECT abstract_id FROM abstracts WHERE isML=1 LIMIT 10000"
    cursor.execute(sq1)

    print("Collecting Abstracts")
    abstracts = set()
    for row in cursor:
        abstracts.add(row[0])
    connection.close()
    pm.build_dep_tree_dict(abstracts)
    pm.save_dep_tree("dep_tree_dict.pickle")
pm.load_rules("seed_rules.json")
pm.learning(iterations=3)
pm.save_rules(filename="learned_rules.json")