import pickle
import os

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)
with open(os.path.join(path_to_mlgenome, "ml_acronyms.pickle"), "rb") as handle:
    acronym_dictionary = pickle.load(handle)


avg_len = 0
max_len = 0

for count, acronym in enumerate(acronym_dictionary):
    def_len = len(acronym_dictionary[acronym])
    avg_len = (count*avg_len + def_len) / (count + 1)

    max_len = max(max_len, def_len)
    #if def_len == 1:
    print(f"{acronym}: {acronym_dictionary[acronym]}")

print(f"Acronyms: {len(acronym_dictionary)}")
print(f"Avg-Length: {avg_len} | Max-Length: {max_len}")