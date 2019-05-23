import random
import pickle
from time import time
import spacy
import os
import numpy as np
from scipy.special import comb
from scipy.optimize import nnls

from src.utils.LoopTimer import LoopTimer

from MLGenome.MLARes.occ import distances


class occTopMention:
    def __init__(self, path):
        with open(path, "rb") as handle:
            occ = pickle.load(handle)

        self.mentions = occ["mentions"]
        self.m_strings = [self.mentions[i]["string"] for i in range(len(self.mentions))]
        self.clusters = occ["cluster_matrix"]
        self.similarity_matrix = occ["similarity_matrix"]

        self.rep2cid = dict()
        self.cid2rep = dict()

        for cid in range(self.clusters.shape[1]):
            if sum(self.clusters[:, cid]) > 0:
                m_ids = np.where(self.clusters[:, cid] == 1)[0]
                top_rep_rating = 0
                top_rep = m_ids[0]
                for v in m_ids:
                    distance = 0
                    for u in m_ids:
                        if v == u:
                            continue

                        d = self.similarity_matrix[v, u]
                        distance += d
                    if len(m_ids) > 1:
                        distance = distance / (len(m_ids) - 1)
                    else:
                        distance = 1
                    if distance > top_rep_rating:
                        top_rep = self.mentions[v]['string']
                        top_rep_rating = distance
                self.cid2rep[cid] = top_rep
                if top_rep not in self.rep2cid:
                    self.rep2cid[top_rep] = list()
                self.rep2cid[top_rep].append(cid)

    def get(self, test_string):
        if test_string.lower() in self.m_strings:
            mid = self.m_strings.index(test_string.lower())

            cluster_ids = np.where(self.clusters[mid, :] == 1)[0]
            top_reps = set()
            for cid in cluster_ids:
                top_rep_rating = 0
                top_rep = test_string
                m_ids = np.where(self.clusters[:, cid] == 1)[0]
                for v in m_ids:
                    rating = 0
                    for u in m_ids:
                        if v == u:
                            continue
                        r = self.similarity_matrix[v, u]
                        rating += r
                    if len(m_ids) > 1:
                        rating = rating / (len(m_ids) - 1)
                    else:
                        rating = 1
                    if rating > top_rep_rating:
                        top_rep = self.mentions[v]['string']
                        top_rep_rating = rating
                top_reps.add(top_rep)
            return list(top_reps)
        else:
            #print(f"out {test_string}")
            return [test_string]


class OverlappingCorrelationClustering:
    def __init__(self, num_clusters, p, mentions, nlp_model):
        path_to_db = "/media/norpheo/mySQL/db/ssorc"
        nlp_path = os.path.join(path_to_db, "models", nlp_model)
        print("[occ.py] - Loading spaCy")
        nlp = spacy.load(nlp_path)

        self.mentions = mentions
        self.num_clusters = num_clusters
        self.p = p
        self.num_mentions = len(mentions)
        self.m_range = range(self.num_mentions)

        label_combs = [comb(self.num_clusters, p_, exact=True) for p_ in range(1, p+1)]
        n_labels = sum(label_combs)

        """
            Compute Similarity Matrix
        """

        expanded_mentions = list()

        for mention in self.mentions:
            expanded_list = [mention['orth_with_ws'].copy()]
            distances.expand_mention(expanded_list)

            expanded_mentions.append(expanded_list)

        """
            For Vectors ONLY
            Removing Ignore_Words
            Compute and Store Vectors
        """

        expanded_mentions_vectors = list()

        lt = LoopTimer(update_after=200, avg_length=20000, target=len(expanded_mentions))
        for expm in expanded_mentions:
            list_strings = ["".join([token for token in tokens if token not in distances.ignore_words]) for tokens in expm]
            list_spacy = [nlp(string) for string in list_strings]
            avg_vec = sum([doc.vector for doc in list_spacy]) / len(list_spacy)
            avg_vec = avg_vec.reshape(1, -1)

            expanded_mentions_vectors.append(avg_vec)

            lt.update("Calc Vectors")

        """
            Compute Vector-Similarity-Matrix
        """
        print()
        lt = LoopTimer(update_after=200, avg_length=20000, target=self.num_mentions * self.num_mentions)
        self.sim_matrix = []
        for vid, v in enumerate(expanded_mentions_vectors):
            sub_sims = []
            for uid, u in enumerate(expanded_mentions_vectors):
                similarity = distances.occ_vec(v, u)
                sub_sims.append(similarity)
                lt.update("Calc Similarities")
            self.sim_matrix.append(sub_sims)

        """
        lt = LoopTimer(update_after=200, avg_length=20000, target=self.num_mentions*self.num_mentions)
        self.sim_matrix = []
        for vid, v in enumerate(expanded_mentions):
            sub_sims = []
            for uid, u in enumerate(expanded_mentions):
                similarity = distances.occ_s(v, u)
                sub_sims.append(similarity)
                lt.update("Calc Similarities")
            self.sim_matrix.append(sub_sims)
        """

        self.sim_matrix = np.array(self.sim_matrix)

        """
            Compute Cluster Initialization
        """

        labeling = dict()
        for mention in self.m_range:
            labeling[mention] = random.randint(0, n_labels-1)

        print()
        lt = LoopTimer(update_after=100, avg_length=2000, target=self.num_mentions)
        self.clusters = []
        for n_mention in self.m_range:
            label_id = labeling[n_mention]
            p = 0
            while label_id >= 0:
                label_id -= label_combs[p]
                p += 1
            label_id += label_combs[p - 1]
            assigned_clusters = []
            calc_p = p
            while calc_p > 1:
                calc_p -= 1
                cluster_cp = int(label_id / label_combs[calc_p - 1])
                label_id -= cluster_cp * label_combs[calc_p - 1]
                assigned_clusters.append(cluster_cp)
            assigned_clusters.append(label_id)
            c_vec = [1 if c in assigned_clusters else 0 for c in range(self.num_clusters)]
            self.clusters.append(c_vec)
            lt.update("Build Cluster Matrix")
        self.clusters = np.array(self.clusters)

        # self.clusters[mention_id , cluster_id]
        print()

    def cost_occ(self):
        cocc_sum = 0
        for v in self.m_range:
            for u in self.m_range:
                if u == v:
                    continue

                jac_coef = self.jac_coef(v, u)
                sim = self.sim_matrix[v, u]

                cocc_sum += abs(jac_coef - sim)

        return cocc_sum*0.5

    def get_set(self, v):
        v_clusters = np.where(self.clusters[v, :] == 1)[0]
        return set(v_clusters)

    def optimize(self, tol=0.00005, save_path = None):

        nnls_t = 0
        build_matrix_t = 0
        get_best_solution_t = 0
        calc_cost_t = 0

        best_cocc = self.cost_occ()

        print(f"Start Optimization with: {best_cocc}")

        init_line = np.ones(self.num_clusters + 1).reshape(1, self.num_clusters + 1)
        init_line[0][0] = -1
        ones_vector = np.ones((self.num_mentions, 1))

        lt = LoopTimer(update_after=1, avg_length=20000)

        best_cluster = np.concatenate(self.clusters)

        while True:
            for v in self.m_range:
                """
                    Construct Matrix
                """
                bm_t_start = time()

                zj_vec = self.sim_matrix[v, :].reshape(self.num_mentions, 1)
                mid_matrix = self.clusters*(ones_vector+zj_vec)
                a_matrix = np.concatenate((np.concatenate((zj_vec, -mid_matrix), axis=1), init_line), axis=0)

                a_matrix = np.delete(a_matrix, v, axis=0)


                n_sj_vec = self.clusters.sum(1).reshape(self.num_mentions, 1) * zj_vec
                b_vec = np.concatenate((-n_sj_vec, np.array([[0]])), axis=0).reshape(self.num_mentions+1)
                b_vec = np.delete(b_vec, v, axis=0)


                bm_t_end = time()
                build_matrix_t += (bm_t_end - bm_t_start)

                """
                    Non Negative Least Squares
                """
                nnls_t_start = time()
                nnls_result = nnls(a_matrix, b_vec)[0][1:]
                nnls_ind = np.argpartition(nnls_result, -self.p)[-self.p:]
                nnls_ind_sorted = nnls_ind[np.argsort(nnls_result[nnls_ind])][::-1]
                nnls_t_end = time()
                nnls_t += (nnls_t_end - nnls_t_start)
                """
                    Retrieve best feasible solution 
                """
                gbs_t_start = time()
                min_dist = float("inf")
                best_sq = float("inf")
                for q in range(1, self.p+1):
                    Sq_ = nnls_ind_sorted[0:q]
                    sq = set(Sq_)
                    dist = 0
                    for j in self.m_range:
                        dist += abs(distances.occ_h(sq, self.get_set(j)) - self.sim_matrix[v, j])

                    if dist < min_dist:
                        min_dist = dist
                        best_sq = np.copy(Sq_)

                for i in range(self.num_clusters):
                    self.clusters[v, i] = 1 if i in best_sq else 0

                gbs_t_end = time()
                get_best_solution_t += (gbs_t_end - gbs_t_start)

            cc_t_start = time()
            new_cocc = self.cost_occ()
            cc_t_end = time()
            calc_cost_t += (cc_t_end-cc_t_start)

            sum_t = calc_cost_t + build_matrix_t + nnls_t + get_best_solution_t
            bmt = round((build_matrix_t / sum_t) * 100, 2)
            nnlst = round((nnls_t/sum_t)*100, 2)
            gbst = round((get_best_solution_t/sum_t) * 100, 2)
            cct = round((calc_cost_t / sum_t) * 100, 2)
            lt.update(f"Optimize: {best_cocc} -> {new_cocc} | Build Matrix: {bmt} % |  NNLS: {nnlst} % | GBS: {gbst} % | CCT: {cct} %")
            #print()
            if abs(best_cocc - new_cocc) < tol or new_cocc > best_cocc:
                break
            best_cocc = new_cocc
            best_cluster = np.copy(self.clusters)
            if save_path is not None:
                self.save(save_path)

        self.clusters = best_cluster
        print()
        print(f"End Optimization with: {best_cocc}")

    def jac_coef(self, v, u):
        v_set = set(np.where(self.clusters[v, :] == 1)[0])
        u_set = set(np.where(self.clusters[u, :] == 1)[0])

        return len(v_set.intersection(u_set)) / len(v_set.union(u_set))

    def print_result(self):
        for cid in range(self.num_clusters):
            m_ids = np.where(self.clusters[:, cid] == 1)[0]
            mentions = [self.mentions[m_id]["string"] for m_id in m_ids]
            if len(mentions) > 1:
                print(f"{cid}: {mentions}")

    def save(self, path):

        save_dict = {"mentions": self.mentions,
                     "cluster_matrix": self.clusters,
                     "similarity_matrix": self.sim_matrix}

        with open(path, 'wb') as handle:
            pickle.dump(save_dict, handle)


