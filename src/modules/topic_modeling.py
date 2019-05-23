import gensim
import pickle
import numpy as np
import scipy.sparse

from MLGenome.MLARes.occ.occ import occTopMention


path_to_db = "/media/norpheo/mySQL/db/ssorc"


class TopicModelingLDA:
    def __init__(self, path_to_info):
        with open(path_to_info, "rb") as handle:
            info = pickle.load(handle)
        self.dictionary = gensim.corpora.Dictionary.load(info["dic_path"])
        with open(info["model_path"], 'rb') as model_file:
            self.model = pickle.load(model_file)

        self.is_merged = info['is_merged']

        if self.is_merged:
            self.otm = occTopMention(path=info['otm_path'])

            self.topic2cluster = dict()
            for topic, comp in enumerate(self.model.components_):
                comp_sum = np.sum(comp)

                top_cid = None
                top_cid_prop = 0
                for cid in self.otm.cid2rep:
                    rep = self.otm.cid2rep[cid]
                    if rep in self.dictionary.token2id:
                        rep_idx = self.dictionary.token2id[rep]
                        prop = comp[rep_idx] / comp_sum

                        if prop > 0.001 and prop > top_cid_prop:
                            top_cid = cid
                            top_cid_prop = prop
                if top_cid is not None:
                    self.topic2cluster[topic] = top_cid

    def num_topics(self):
        return self.model.components_.shape[0]

    def get_topic_dist(self, doc):
        if self.is_merged:
            for ent in doc.ents:
                ent.merge(ent.root.tag_, ent.orth_, ent.label_)

            tokens = []
            for token in doc:
                append_tokens = self.otm.get(token.text)
                for append_token in append_tokens:
                    tokens.append(append_token)
        else:
            tokens = [token.text for token in doc]

        bow = self.dictionary.doc2bow(tokens)
        row = []
        col = []
        data = []

        for entry in bow:
            row.append(0)
            col.append(entry[0])
            data.append(entry[1])

        m = 1
        n = len(self.dictionary)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)

        feature_vector = scipy.sparse.csr_matrix((data, (row, col)), shape=(m, n)).toarray()
        topic_distribution = self.model.transform(feature_vector)

        return topic_distribution[0]

    def get_topic(self, doc):

        topic_dist = self.get_topic_dist(doc=doc)
        topic = topic_dist.argmax()
        return topic

    def get_cluster(self, doc):
        topic = self.get_topic_dist(doc)

        if topic in self.topic2cluster:
            return self.topic2cluster[topic]
        else:
            return None
