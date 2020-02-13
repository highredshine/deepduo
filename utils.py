import numpy as np

# WINDOW_SIZE is the maximum length of all sentences found in this project. Used for padding later.
MAX_TOKEN_SIZE = 14
# Padding token
PAD_TOKEN = "*PAD*"


def cast_to_float(string):
    """
    change string to numeric for features like day and time
    """
    if string == 'null':
        return float(0)
    else:
        return float(string)


def cast_to_int(string):
    """
    change string to numeric for features like dependency_edge_head
    """
    if string == 'null':
        return 0
    else:
        return int(string)


def create_feature_to_id(metadata_list, feature_num):
    """
    a generic function to create a dictionary of feature_to_id mapping for a feature.
    Parameters:
        metadata_list: the train dataset to create the mapping from.
        feature_num: index from a list of metadata for a token that represents a feature
    Returns:
        feature_to_id: a dictionary of mapping
    """
    all_features = set()
    # for each exercise's metadata matrix:
    for metadata in metadata_list:
        # for each metadata of a word in an exercise:
        for m in metadata:
            all_features.add(m[feature_num])
    feature_to_id = {}
    for i, feature in enumerate(all_features):
        feature_to_id[feature] = i
    return feature_to_id

#count is the starter, which builds upon previous dataset.
def create_meta_feature_to_id(metadata_list, feature_num, count):
    all_features = set()
    for metadata in metadata_list:
        for m in metadata:
            all_features.add(m[feature_num])
    feature_to_id = {}
    for i, feature in enumerate(all_features):
        feature_to_id[feature] = i + count
    return feature_to_id, i + count 


def create_token_to_idx(sentence_list):
    """The function returns a dict mapping the tokent to idx; The 0th index is padding
    Parameters:
        sentences_list: list of sentences(exericses), each of which is also a list that contains tokens
    Returns:
        dict -- token to idx mapping with an additional token
    """
    all_tokens = set()
    # for each sentence
    for sent in sentence_list:
        # add all unique tokens
        for token in sent:
            all_tokens.add(token)
    token_to_idx = {}
    token_to_idx[PAD_TOKEN] = 0
    i = 0
    for token in all_tokens:
        if token not in token_to_idx:
            i += 1
            token_to_idx[token] = i
    return token_to_idx


def process_sentence(raw_sent, token_to_idx):
    """The function is used to convert raw sentence to idx for training/testing the models
    Arguments:
        raw_sent {np.array} -- list of list of token
        token_to_idx {dict} -- dict mapping from token to idx
    """
    sent_idx = np.zeros(raw_sent.shape, dtype=int)
    for i, sent in enumerate(raw_sent):
        sent_idx[i] = np.array([token_to_idx[token] for token in sent])
    return sent_idx
