import os
import time
import datetime
import numpy as np

from langdetect import detect_langs  # https://pypi.org/project/langdetect/
from langdetect.lang_detect_exception import LangDetectException

from timeit import default_timer as timer
from sklearn.externals import joblib
import sklearn.metrics as skmetrics


def check_string_for_english(input_text):
    try:
        detection = detect_langs(input_text)

        result = False
        for lang in detection:
            if lang.lang == 'en':
                if lang.prob > 0.9:
                    result = True

        return result
    except LangDetectException:
        return False


def check_array_for_english(input_text):
    text_string = " ".join(input_text)

    return check_string_for_english(text_string)


def makeBigrams(token_array):
    bigram = []

    for index in range(0, len(token_array) - 1):
        bigram.append(str(token_array[index]) + ' ' + str(token_array[index + 1]))
    return bigram


def posFilterString(word_list, pos_list):
    allowed_pos_tags = ['CC', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
                        'POS',
                        'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                        'WDT', 'WP', 'WP$', 'WRB']

    word_list_cleaned = []
    pos_list_cleaned = []
    if len(word_list) == len(pos_list):
        for i in range(0, len(pos_list)):
            if pos_list[i] in allowed_pos_tags:
                pos_list_cleaned.append(pos_list[i])
                word_list_cleaned.append(word_list[i])
    return word_list_cleaned, pos_list_cleaned


def add_vector_to_sparse_matrix(matrix, vector, offset):
    for elem in vector:
        matrix[0, int(elem[0]) + offset] = float(elem[1])


def add_string_to_sparse_matrix(matrix, input_string, offset):
    sparse_vec = get_vec_tuple_from_string(input_string)

    add_vector_to_sparse_matrix(matrix, sparse_vec, offset)


def get_vec_tuple_from_string(input_string):
    print(input_string)
    return [tuple(t.split(":")) for t in input_string.split()]


def append_vec2data(sparse_vec, data, row, col, row_count, col_offset):
    for entry in sparse_vec:
        row.append(row_count)
        col.append(int(entry[0]) + col_offset)
        data.append(float(entry[1]))


def measure(m_function, text):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('[%H:%M:%S]')
    print(st + " [Starting] (" + text + ")")
    start = timer()
    fresult = m_function()
    end = timer()
    result_time = end - start
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('[%H:%M:%S]')
    print(st + " [Finished] (" + text + ") after " + str(result_time) + " seconds")
    return fresult


def load_sk_model(filename):
    dirname = os.path.dirname(__file__)
    model_dir = os.path.join(dirname, '../../models')
    model_file = os.path.join(model_dir, filename)

    if os.path.isfile(model_file):
        model = joblib.load(model_file)
        return model
    else:
        print(filename + " not found")


def save_sk_model(model, filename):
    dirname = os.path.dirname(__file__)
    model_dir = os.path.join(dirname, '../../models')
    model_file = os.path.join(model_dir, filename)

    joblib.dump(model, model_file)
    print('Model ' + filename + ' saved')


def lookahead(iterable):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False).
    """
    # Get an iterator and pull the first value.
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield last, True
        last = val
    # Report the last value.
    yield last, False


def print_scoring(targets, prediction, avg_type="weighted"):
    labels = list(set(targets))
    precision = dict()
    recall = dict()

    conf_matrix = skmetrics.confusion_matrix(targets, prediction, labels=labels)
    prec_score = skmetrics.precision_score(targets, prediction, labels=labels, average=avg_type)
    accuracy_score = skmetrics.accuracy_score(targets, prediction)
    recall_score = skmetrics.recall_score(targets, prediction, average=avg_type)
    f1_score = skmetrics.f1_score(targets, prediction, average=avg_type)

    print("Occurences of Labels (Relevant | Retrieved)")
    for label in labels:
        label_idx = labels.index(label)

        rel = np.sum(conf_matrix, axis=1)[label_idx]
        ret = np.sum(conf_matrix, axis=0)[label_idx]

        precision[label] = conf_matrix[label_idx][label_idx] / ret if ret > 0 else 0
        recall[label] = conf_matrix[label_idx][label_idx] / rel

        print(f"{label}: {rel} | {ret}")

    print()
    print("     \tPreci | Recall")
    for label in labels:
        print(f"{str(label)[0:5]}:\t{str(precision[label])[0:5]} | {str(recall[label])[0:5]}")

    print()
    print(f"Accuracy:  {accuracy_score}")
    print(f"Precision: {prec_score}")
    print(f"Recall:    {recall_score}")
    print(f"F1-Score:  {f1_score}")
