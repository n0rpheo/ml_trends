import difflib
from scipy.spatial.distance import cosine

"""
    Context Independent Features
    ============================
"""


# String Ähnlichkeit
def similarity_ratio(algo1, algo2):
    return difflib.SequenceMatcher(None, algo1, algo2).ratio()


def is_prefix(mention1, mention2):
    mention_len = min(len(mention1), len(mention2))

    for i in range(mention_len):
        if mention1[i] != mention2[i]:
            return False
    return True


def is_suffix(mention1, mention2):
    if len(mention1) >= len(mention2):
        long_mention = mention1
        short_mention = mention2
    else:
        long_mention = mention2
        short_mention = mention1
    n_long = len(long_mention)
    n_short = len(short_mention)

    for i in range(n_long-n_short, n_long):
        if short_mention[i+n_short-n_long] != long_mention[i]:
            return False
    return True


def is_infix(mention1, mention2):
    if len(mention1) >= len(mention2):
        long_mention = mention1
        short_mention = mention2
    else:
        long_mention = mention2
        short_mention = mention1
    n_long = len(long_mention)
    n_short = len(short_mention)

    for i in range(n_long-n_short):
        if short_mention[0] == long_mention[i]:
            for j in range(n_short):
                if short_mention[j] != long_mention[i+j]:
                    break
            else:
                return True
    return False


def intersection_of_words(tokens_algo1, tokens_algo2):
    set_algo1 = set(tokens_algo1)
    set_algo2 = set(tokens_algo2)

    intersection = set_algo1.intersection(set_algo2)
    return (1.0 + len(intersection)) / (1.0 + max(len(set_algo1), len(set_algo2)))


def vec_sim(vec1, vec2):
    return 1 - cosine(vec1, vec2)


def make_features(mention, candidate, col, row, data_array, row_count):
    m_string = mention["string"]
    m_orth = mention["orth"]
    m_lemma = mention['lemma']
    m_pos = mention['pos']
    m_length = mention['length']
    m_swc = mention['starts_with_capital']
    m_is_acronym = mention['is_acronym']

    c_string = candidate["string"]
    c_orth = candidate["orth"]
    c_lemma = candidate['lemma']
    c_pos = candidate['pos']
    c_length = candidate['length']
    c_swc = candidate['starts_with_capital']
    c_is_acronym = candidate['is_acronym']


    similarity = similarity_ratio(m_string, c_string)
    intersection = intersection_of_words(m_lemma, c_lemma)

    is_pfix = is_prefix(m_lemma, c_lemma)
    is_ifix = is_infix(m_lemma, c_lemma)
    is_sfix = is_suffix(m_lemma, c_lemma)


    col_count = 0


    row.append(row_count)
    col.append(col_count)
    data_array.append(similarity)
    col_count += 1

    row.append(row_count)
    col.append(col_count)
    data_array.append(intersection)
    col_count += 1

    row.append(row_count)
    col.append(col_count)
    data_array.append(is_pfix)
    col_count += 1

    row.append(row_count)
    col.append(col_count)
    data_array.append(is_ifix)
    col_count += 1

    row.append(row_count)
    col.append(col_count)
    data_array.append(is_sfix)
    col_count += 1





