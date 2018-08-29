from src.modules.topic_modeling import TopicModeling


tm = TopicModeling('ssorc')

print("Build Feature File")
tm.build_features(prefix='lemma_10k', num_samples=10000)

print("Load Features")
#tm.load_features(prefix='lemma_10k')

print("Train Model")
#tm.train(num_topics=500, passes=3, model_name="lda_500_topics.model")
