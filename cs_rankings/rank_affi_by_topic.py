import os
import pandas as pd

from src.utils.LoopTimer import LoopTimer

path_to_db = "/media/norpheo/mySQL/db/ssorc"
author_ratings = pd.read_pickle(os.path.join(path_to_db, 'csrankings', 'ratings.pd'))
author_db = pd.read_pickle(os.path.join(path_to_db, 'pandas', 'author_db.pandas'))

fields = dict()
fields["nn_ids"] = [1, 95, 207, 268, 338]
fields["naive_bayes_ids"] = [22, 357]
fields["ql_ids"] = [31]
fields["adaboost_ids"] = [75]
fields["clustering_ids"] = [78, 497]
fields["dt_ids"] = [81, 392, 435]
fields["hmm_ids"] = [160]
fields["pca_ids"] = [168]
fields["svm_ids"] = [245, 393]
fields["nlp_ids"] = [271, 273, 492]
fields["lr_ids"] = [345]
fields["crf_ids"] = [410]

with open(os.path.join(path_to_db, "csrankings", "csrankings.csv"), "r") as affi_file:
    affi_file.readline()
    names = list()
    affiliations = list()

    for line in affi_file:
        tokens = line.split(",")
        name = tokens[0]
        affiliation = tokens[1]
        names.append(name)
        affiliations.append(affiliation)

affiliationDF = pd.DataFrame(affiliations, index=names, columns=["affiliation"])
affiliation_list = list(set(affiliationDF["affiliation"].tolist()))

field = "svm_ids"

a_list = list()
rating_list = list()

lt = LoopTimer(update_after=1, avg_length=20, target=len(affiliation_list))
for affiliation in affiliation_list:
    affi_authors = affiliationDF.loc[affiliationDF["affiliation"] == affiliation].index.tolist()

    affiliation_author_ids = list()
    for affi_author in affi_authors:
        if len(author_db.loc[author_db["authors"] == affi_author]) > 0:
            author_name = author_db.loc[author_db["authors"] == affi_author]["authors"].tolist()[0]
            author_id = author_db.loc[author_db["authors"] == affi_author].index.tolist()[0]
            affiliation_author_ids.append(author_id)

    field_rating = author_ratings[fields[field]].sum(axis=1)[affiliation_author_ids].sum()

    a_list.append(affiliation)
    rating_list.append(field_rating)

    i = lt.update("Rate Affiliation")

affiliationDF = pd.DataFrame(rating_list, index=a_list, columns=["rating"])

print()
print(field)
print(affiliationDF.sort_values(by="rating", ascending=False))