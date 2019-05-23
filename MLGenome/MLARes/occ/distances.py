import difflib
import pickle
import os

from sklearn.metrics.pairwise import cosine_similarity

path_to_db = "/media/norpheo/mySQL/db/ssorc"
nlp_model = "en_wa_v2"

path_to_mlgenome = os.path.join(path_to_db, "mlgenome", nlp_model)

with open(os.path.join(path_to_mlgenome, "ml_acronyms.pickle"), "rb") as handle:
    acronym_dict = pickle.load(handle)

ignore_words = ['global',
                'development',
                'local',
                'structure',
                'standard',
                'simulation',
                'stochastic',
                'model',
                'models',
                'analysis',
                'classifier',
                'classifiers',
                'method',
                'problem',
                'problems']


def comb_lists(list1):
    # store all the sublists
    sublist = list()

    # first loop
    for i in range(len(list1) + 1):

        # second loop
        for j in range(i + 1, len(list1) + 1):
            # slice the subarray
            sub = (i, j)
            sublist.append(sub)

    return sublist


def expand_mention(expanded_list):
    for tlid, token_list in enumerate(expanded_list):
        candidate_list = comb_lists(token_list)

        best_candidate = None
        len_best_candidate = 0
        best_cstring = None

        for cid, candidate in enumerate(candidate_list):
            c_string = "".join(token_list[candidate[0]:candidate[1]]).strip()
            len_candidate = candidate[1] - candidate[0]
            if c_string in acronym_dict and len_candidate > len_best_candidate:
                best_candidate = candidate
                best_cstring = c_string
                len_best_candidate = len_candidate

        if best_candidate is not None:
            longforms = [lf['orth_with_ws'] for lf in acronym_dict[best_cstring]]
            expanded_token_list = list()
            for longform in longforms:
                temp_tl = token_list.copy()
                temp_tl[best_candidate[0]:best_candidate[1]] = longform
                expanded_token_list.append(temp_tl)
            expanded_list[tlid:tlid+1] = expanded_token_list
            expand_mention(expanded_list)


def occ_vec(u, v):

    return cosine_similarity(u, v)[0][0]


def occ_s(u, v):
    u_list_orth = [[token.strip() for token in tokens if token.strip() not in ignore_words] for tokens in u]
    u_list_strings = ["".join(tokens) for tokens in u_list_orth]

    v_list_orth = [[token.strip() for token in tokens if token.strip() not in ignore_words] for tokens in v]
    v_list_strings = ["".join(tokens) for tokens in v_list_orth]

    """
        String Rating
    """

    string_rating = 0
    count = 0
    for vs in v_list_strings:
        for us in u_list_strings:
            vu_r = difflib.SequenceMatcher(None, vs, us).ratio()
            uv_r = difflib.SequenceMatcher(None, us, vs).ratio()
            rating = (vu_r + uv_r) / 2
            count += 1
            string_rating = ((count - 1) * string_rating + rating) / count

    """
        Same Token Rating
    """
    token_rating = 0
    count = 0
    for vo in v_list_orth:
        vo_set = set(vo)
        for uo in u_list_orth:
            uo_set = set(uo)
            n_ident_tokens = len(vo_set.intersection(uo_set))
            n_longer_token = max(len(uo_set), len(vo_set))
            rating = n_ident_tokens / n_longer_token
            count += 1
            token_rating = ((count - 1) * token_rating + rating) / count

    similarity = (string_rating + token_rating) / 2
    return similarity


def occ_h(e_set, f_set):
    return len(e_set.intersection(f_set)) / len(e_set.union(f_set))