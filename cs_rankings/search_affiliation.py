import os
import pandas as pd

path_to_db = "/media/norpheo/mySQL/db/ssorc"
author_db = pd.read_pickle(os.path.join(path_to_db, 'pandas', 'author_db.pandas'))

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

df = pd.DataFrame(affiliations, index=names, columns=["affiliation"])

affiliation = "Munich"
affi_authors = df.loc[df["affiliation"] == affiliation].index.tolist()

schools = list(set(df["affiliation"].tolist()))

for school in schools:
    if affiliation in school:
        print(school)