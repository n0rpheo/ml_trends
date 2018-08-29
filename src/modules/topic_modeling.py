import os
import gensim

import src.features.tm_features as tmf


class TopicModeling:
    def __init__(self, dtype):
        dirname = os.path.dirname(__file__)

        dic_dir = os.path.join(dirname, '../../data/processed/' + dtype + '/dictionaries')
        lemma_path = os.path.join(dic_dir, 'lemma.dic')

        tfidf_path = os.path.join(dirname, '../../data/processed/' + dtype + '/tfidf')
        tfidf_lemma_path = os.path.join(tfidf_path, 'lemma_model.tfidf')

        self.model_dir = os.path.join(dirname, '../../models/' + dtype)

        self.dtype = dtype
        self.lemma_dic = gensim.corpora.Dictionary.load(lemma_path)
        self.lemma_tfidf = gensim.models.TfidfModel.load(tfidf_lemma_path)

        self.tm_features = None
        self.corpus_tfidf = None
        self.lda_model = None

    def load_features(self, prefix='lemma'):
        self.tm_features = tmf.get_scipy_features(self.dtype, prefix=prefix)
        if self.tm_features is None:
            print("File not Found. Build Feature File first.")
        else:
            print("Features loaded.")

    def build_features(self, prefix='lemma', num_samples=10000):
        tmf.build_scipy_feature_file(dtype=self.dtype, prefix=prefix, num_samples=num_samples)
        print()
        print("Features built.")

    def load_model(self, model_name):
        model_path = os.path.join(self.model_dir, model_name)

        if os.path.isfile(model_path):
            self.lda_model = gensim.models.LdaMulticore.load(model_path)
            print('Model loaded')
        else:
            print('Model not loaded')

    def load_corpus(self, prefix=''):
        if self.corpus_tfidf is None:
            if self.tm_features is not None:
                self.corpus_tfidf = gensim.matutils.Sparse2Corpus(self.tm_features.transpose())
            else:
                self.load_features(prefix=prefix)
                if self.tm_features is not None:
                    self.corpus_tfidf = gensim.matutils.Sparse2Corpus(self.tm_features.transpose())
                else:
                    print("Corpus couldn't be loaded. Built Feature File first.")

    def train(self, prefix='lemma', num_topics=100, update_every=1, passes=1, model_name="lda.model"):
        self.load_corpus(prefix=prefix)

        print('Training Model.')
        self.lda_model = gensim.models.LdaMulticore(corpus=self.corpus_tfidf,
                                                    id2word=self.lemma_dic,
                                                    num_topics=num_topics,
                                                    eval_every=update_every,
                                                    passes=passes,
                                                    workers=5)
        print('Model Trained.')

        model_path = os.path.join(self.model_dir, model_name)

        self.lda_model.save(model_path)
        print('Model Saved.')

    def get_topic_dist(self, input_lemmas):
        lemma_bow = self.lemma_dic.doc2bow(input_lemmas)
        vec_lemma_tfidf = self.lemma_tfidf[lemma_bow]
        vec_lda = self.lda_model[vec_lemma_tfidf]

        return vec_lda

    def get_topic(self, input_lemmas):
        topic_dist = self.get_topic_dist(input_lemmas)

        topic = None
        max_p = 0
        for entry in topic_dist:
            topic_id = entry[0]
            topic_p = entry[1]
            if topic_p > max_p:
                max_p = topic_p
                topic = topic_id

        return topic

    def phi(self, topic, word):
        if isinstance(word, str):
            word_str = word
            if word_str not in self.lemma_dic.token2id:
                return 0.0
        elif isinstance(word, int):
            word_str = self.lemma_dic[word]
        else:
            return 0.0

        word_dist = self.lda_model.show_topic(topic, topn=len(self.lemma_dic))
        for word_prob in word_dist:
            if word_prob[0] == word_str:
                return word_prob[1]

        return 0.0
