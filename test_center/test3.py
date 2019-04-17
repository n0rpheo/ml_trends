import os
import spacy









lst = [1, 2, None, None, None, 5, 6, 7]
print(lst)
for i in range(lst.count(None)):
    lst.remove(None)

print(lst)
exit()




path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp = spacy.load(os.path.join(path_to_db, "models", "en_core_web_sm_nertrained"))
vocab = nlp.vocab.from_disk(os.path.join(path_to_db, "dictionaries", "ner_spacy.vocab"))

nlp = spacy.load("en_core_web_sm")
#apple = nlp.vocab.strings["apple"]

doc = nlp("Building a computational model for how the visual cortex identifies objects is a problem that has attracted much attention over the years. Generally, the interest has been in creating models that are translation, rotation, and luminance invariant. In this paper, we utilize the philosophy of Hough Transform to create a model for detecting straight lines under conditions of discontinuity and noise. A neural network that can learn to perform a Hough Transform-like computation in an unsupervised manner is the main takeaway from this work. Performance of the network when presented with straight lines is compared with that of human subjects. Optical illusions like the Poggendorff illusion could potentially find an explanation in the framework of our model.")

#print(apple)
#print(nlp.vocab.strings[399])
#print(nlp.vocab.strings[7972625988311165362])

for token in doc:
    print(f"{token.head} -> {token.dep_} ({token.dep}) -> {token.text}")