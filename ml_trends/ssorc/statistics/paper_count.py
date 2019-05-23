import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_trends.ssorc.statistics import jourven_list


path_to_db = "/media/norpheo/mySQL/db/ssorc"
path_to_fig_save = "/media/norpheo/Daten/Masterarbeit/thesis/Results/fig"

df = pd.read_pickle(os.path.join(path_to_db, 'annotations_version', 'en_core_web_sm_nertrained_v3', 'info_db.pandas'))
print(len(df))

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

all_year_dict = dict()
for year in range(1948, 2019):
    all_year_dict[year] = 0

category_ts = dict()
for category in jourven_list.journals:
    category_ts[category] = dict()
    for year in range(1948, 2019):
        category_ts[category][year] = 0

abs_cats = dict()

for abstract_id, entry in df.iterrows():
    year = entry['year']
    journal = entry['journal']
    venue = entry['venue']

    if journal in journals or venue in venues:
        all_year_dict[year] += 1

    for cat in category_ts:
        if journal in jourven_list.journals[cat] or venue in jourven_list.venues[cat]:
            category_ts[cat][year] += 1

            if abstract_id not in abs_cats:
                abs_cats[abstract_id] = set()

            abs_cats[abstract_id].add(cat)

cat_n_grams_list = list()
for aid in abs_cats:
    ngram = ", ".join(abs_cats[aid])
    cat_n_grams_list.append(ngram)

cat_n_grams_set = set(cat_n_grams_list)
print(len(cat_n_grams_set))
for ngram in cat_n_grams_set:
    print(f"{ngram}: {cat_n_grams_list.count(ngram)}")

"""
    Plot for every Category
"""
bar_width = 0.5
opacity = 0.8
for cat in category_ts:

    year_dict = category_ts[cat]

    category_string = cat.replace(" ", "")

    years_list = [key for key in year_dict.keys() if key < 2018]
    years_list.sort()

    x = np.array(years_list)
    y = np.array([year_dict[year] for year in years_list])

    fig, ax = plt.subplots()
    #ax.plot(x, y)
    ax.bar(x, y, bar_width, alpha=opacity, color='b')
    start, end = ax.get_xlim()
    start = int(start)
    end = int(end)
    ax.xaxis.set_ticks(np.arange(start, end, 2))
    # ax.set_yscale("log", nonposy='clip')
    plt.xlim(start, 2020)
    plt.xticks(rotation=70)
    plt.xlabel("Year")
    plt.ylabel("Publications")

    plt.savefig(os.path.join(path_to_fig_save, f"cat_{category_string}_pub_count.png"))
    # plt.show()

"""
    Plot stacked Bar
"""

stacked_fig, stacked_ax = plt.subplots()
stacked_y = None
plots = list()
header = list()
for cat in category_ts:

    year_dict = category_ts[cat]

    category_string = cat.replace(" ", "")

    years_list = [key for key in year_dict.keys() if key < 2018]
    years_list.sort()

    x = np.array(years_list)
    y = np.array([year_dict[year] for year in years_list])

    if stacked_y is None:
        header.append(cat)
        plots.append(stacked_ax.bar(x, y, bar_width, alpha=opacity))
        stacked_y = y
    else:
        header.append(cat)
        plots.append(stacked_ax.bar(x, y, bar_width, alpha=opacity, bottom=stacked_y))
        stacked_y += y
bar_width = 0.5
opacity = 1
start, end = stacked_ax.get_xlim()
start = int(start)
end = int(end)
stacked_ax.xaxis.set_ticks(np.arange(start, end, 2))
plt.xlim(start, 2020)
plt.legend([p[0] for p in plots], [h for h in header])
plt.xticks(rotation=70)
plt.xlabel("Year")
plt.ylabel("Publications")
plt.savefig(os.path.join(path_to_fig_save, f"stacked_pub_count.png"))


"""
    Plot All
"""

years_list = [key for key in all_year_dict.keys() if key < 2018]
years_list.sort()
x = years_list
y = [all_year_dict[year] for year in years_list]

bar_width = 0.5
opacity = 0.8

fig, ax = plt.subplots()
#ax.plot(x, y)
ax.bar(x, y, bar_width, alpha=opacity, color='b')

start, end = ax.get_xlim()
start = int(start)
end = int(end)
ax.xaxis.set_ticks(np.arange(start, end, 2))
# ax.set_yscale("log", nonposy='clip')
plt.xlim(start, 2020)
# plt.ylim(0, 1200)
# plt.show()
plt.xticks(rotation=70)
plt.xlabel("Year")
plt.ylabel("Publications")
plt.savefig(os.path.join(path_to_fig_save, f"pub_count.png"))

ax.set_yscale("log", nonposy='clip')
plt.savefig(os.path.join(path_to_fig_save, f"pub_log_count.png"))