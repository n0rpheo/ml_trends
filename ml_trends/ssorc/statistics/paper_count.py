import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_trends.ssorc.statistics import jourven_list


path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_fig_save = "/media/norpheo/Daten/Masterarbeit/thesis/Results/fig"

df = pd.read_pickle(os.path.join(path_to_db, 'annotations_version', 'en_core_web_sm_nertrained_v3', 'info_db.pandas'))


journals = list()
venues = list()

for region in jourven_list.venues:
    vr = jourven_list.venues[region]
    for mag in vr:
        venues.append(mag)
for region in jourven_list.journals:
    vr = jourven_list.journals[region]
    for mag in vr:
        journals.append(mag)

year_dict_by_journal = dict()
for journal in journals:
    year_dict_by_journal[journal] = dict()
    for year in range(1950, 2018):
        year_dict_by_journal[journal][year] = 0

year_dict_by_venue = dict()
for venue in venues:
    year_dict_by_venue[venue] = dict()

    for year in range(1950, 2018):
        year_dict_by_venue[venue][year] = 0

all_year_dict = dict()


for abstract_id, entry in df.iterrows():
    year = entry['year']
    journal = entry['journal']
    venue = entry['venue']

    if journal in journals or venue in venues:
        if year not in all_year_dict:
            all_year_dict[year] = 0
        all_year_dict[year] += 1

    if journal in journals:
        if year not in year_dict_by_journal[journal]:
            year_dict_by_journal[journal][year] = 0
        year_dict_by_journal[journal][year] += 1

    if venue in venues:
        if year not in year_dict_by_venue[venue]:
            year_dict_by_venue[venue][year] = 0
        year_dict_by_venue[venue][year] += 1

"""
for journal in journals:
    if journal not in year_dict_by_journal:
        continue
    year_dict = year_dict_by_journal[journal]

    years_list = [key for key in year_dict.keys() if key < 2018]
    years_list.sort()


    x = years_list
    y = [year_dict[year] for year in years_list]

    fig, ax = plt.subplots()
    ax.plot(x, y)

    start, end = ax.get_xlim()
    start = int(start)
    end = int(end)
    ax.xaxis.set_ticks(np.arange(start, end, 2))
    # ax.set_yscale("log", nonposy='clip')
    plt.xticks(rotation=90)
    plt.xlim(start, 2020)
    # plt.ylim(0, 1200)

    # plt.show()
    plt.savefig(os.path.join(path_to_fig_save, f"journal_{journal}_pub_count.png"))


for venue in venues:
    if venue not in year_dict_by_venue:
        continue
    year_dict = year_dict_by_venue[venue]

    years_list = [key for key in year_dict.keys() if key < 2018]
    years_list.sort()


    x = years_list
    y = [year_dict[year] for year in years_list]

    fig, ax = plt.subplots()
    ax.plot(x, y)

    start, end = ax.get_xlim()
    start = int(start)
    end = int(end)

    ax.xaxis.set_ticks(np.arange(start, end, 2))
    # ax.set_yscale("log", nonposy='clip')
    plt.xticks(rotation=90)
    plt.xlim(start, 2020)
    # plt.ylim(0, 1200)

    # plt.show()
    plt.savefig(os.path.join(path_to_fig_save, f"venue_{venue}_pub_count.png"))
"""


years_list = [key for key in all_year_dict.keys() if key < 2018]
years_list.sort()
x = years_list
y = [all_year_dict[year] for year in years_list]

fig, ax = plt.subplots()
ax.plot(x, y)

start, end = ax.get_xlim()
start = int(start)
end = int(end)
ax.xaxis.set_ticks(np.arange(start, end, 2))
# ax.set_yscale("log", nonposy='clip')
plt.xticks(rotation=90)
plt.xlim(start, 2020)
# plt.ylim(0, 1200)
# plt.show()
plt.savefig(os.path.join(path_to_fig_save, f"aiml_2_pub_count.png"))


fig, ax = plt.subplots()
ax.plot(x, y)

start, end = ax.get_xlim()
start = int(start)
end = int(end)
ax.xaxis.set_ticks(np.arange(start, end, 2))
ax.set_yscale("log", nonposy='clip')
plt.xticks(rotation=90)
plt.xlim(start, 2020)
# plt.ylim(0, 1200)

# plt.show()
plt.savefig(os.path.join(path_to_fig_save, f"aiml_2_pub_log_count.png"))