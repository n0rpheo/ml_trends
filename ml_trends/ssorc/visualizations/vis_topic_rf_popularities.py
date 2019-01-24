import matplotlib.pyplot as plt
import os
import pickle

path_to_db = "/media/norpheo/mySQL/db/ssorc"
df_file_path = os.path.join(path_to_db, "popularities", "df_500topics_dist_hl.pickle")

with open(df_file_path, "rb") as pif:
    dataFrames = pickle.load(pif)

num_topics = len(dataFrames)

nn_topic_list = [8, 74, 198, 211, 218, 310, 456, 494]
svm_topic_list = [4, 287, 345]
dt_topic_list = [387, 420, 426]
lr_topic_list = [229]
kernel_topic_list = [236]

sum_dt = None

plotDF = None

for topic in nn_topic_list:
    df = dataFrames[topic]
    rf_labels = [key for key in df][:-2]

    for rf_label in rf_labels:
        df[rf_label] = df[rf_label] / df['spy']

    if plotDF is None:
        plotDF = df[rf_labels]
    else:
        plotDF = plotDF + df[rf_labels]

    #df = None

    df['sum'] = df['sum'] / df['spy']
    # df['sum'].plot()

    df[rf_labels].plot(kind='bar', stacked=True)

#plotDF.plot(kind='bar', stacked=True)
plt.show()
