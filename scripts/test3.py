import os
from src.utils.path_selection import select_path_from_dir

path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_dic = os.path.join(path_to_db, "dictionaries")

path = select_path_from_dir(path_to_dic)

print(path)