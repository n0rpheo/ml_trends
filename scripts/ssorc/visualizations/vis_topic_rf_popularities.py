import matplotlib.pyplot as plt
import os
import pickle

path_to_db = "/media/norpheo/mySQL/db/ssorc"
df_file_path = os.path.join(path_to_db, "popularities", "df_50topics_dist.pickle")

with open(df_file_path, "rb") as pif:
    dataFrames = pickle.load(pif)

num_topics = len(dataFrames)


for topic in range(15):
    df = dataFrames[topic]
    rf_labels = [key for key in df][:-2]

    for rf_label in rf_labels:
        df[rf_label] = df[rf_label] / df['spy']
    df['sum'] = df['sum'] / df['spy']
    #df[rf_labels].plot(kind='bar', stacked=True)
    df['sum'].plot()
    plt.show()