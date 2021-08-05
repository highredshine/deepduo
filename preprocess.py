import numpy as np
from utils import MAX_TOKEN_SIZE, PAD_TOKEN, cast_to_float, cast_to_int, create_feature_to_id, create_meta_feature_to_id
import copy

def read_key(file_name):
    """
    This imports a 'key' file and reads the label for a corresponding instance_id
    Parameters:
        file_name: the filename pointing to the key file
    Returns:
        labels: a dictionary that maps an instance_id to its label
    """
    labels = {}
    with open(file_name, 'rt') as data_file:
        for line in data_file:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                line = line.split()
            instance_id = line[0]
            label = cast_to_int(line[1])
            labels[instance_id] = label
    return labels


def read_data(file_name, key_file=None):
    """
    This reads the dataset and splits into numpy arrays of sentences, metadata, and labels (along with instance_ids as keys)
    Parameters:
        file_name: file name of the dataset that contains the words and the metadata
        key_file: key file name if there is a separate file for labels. Use read_key function for this option.
    Returns:
        sentence_list: a list of each exercise, when one exercise represents one sentence. shape (num_of_exercises,max_token_size)
        metadata_list: a list of metadata of each exercise. shape (num_of_exercises,num_of_words,num_of_features)
        instance_ids_list: a list of instance_ids for each exercise, used for finding labels. shape (num_of_exercises, num_of_words)
        labels: a dict of labels that maps an instance_id to its label.
    """
    sentence_list = []
    metadata_list = []
    if key_file is None:
        labels = {}
    instance_ids_list = []
    with open(file_name, 'rt') as data_file:
        # sentence represents one exercise, and one exercise is a list of words. shape: (1,max_token_size)
        sentence = []
        # metadata is a list of metadata for each instance (word) of one exrecise. (max_token_size, num_of_features)
        metadata = []
        # exercise_metadata and instance_metadata represent exercise-level and instance-level features.
        # they get concatenated to represent one metadata for each instance, each of which forms a row for metadata matrix.
        exercise_metadata = []
        instance_metadata = []
        # instance_ids is a list of ids of all instances in an exercise.
        instance_ids = []
        if key_file is None:
            labels = {}
        for line in data_file:
            line = line.strip()
            # When a blank line is read, then it means one exercise is finished.
            if len(line) == 0:
                # Sentence Section
                # padding: To have equal tensor sizes, we pad the sentence to the window size
                sentence += [PAD_TOKEN] * (MAX_TOKEN_SIZE - len(sentence))
                sentence_list.append(sentence)
                # renew variable
                sentence = []
                # Metadata Section
                # this function only reads and appends data. 
                # Padding requires more code so separated into another section (cleaner code)
                metadata_list.append(metadata)
                metadata = []
                exercise_metadata = []
                # Label Section
                # add instance_ids
                instance_ids_list.append(instance_ids)
                instance_ids = []
            # For each exercise, some exercise-level features are listed at the start, 
            # with these lines specified with '#' symbol at the start.
            elif line[0] == '#':
                # the prompt (the original language) is not useful for out model. We thus continue.
                if 'prompt' in line:
                    continue
                # We extract all important features.
                else:
                    # split to features (user, countries, days, client, session, format, time)
                    line = line[2:].split()
                    # add all features with the order above (we map to indices later, for cleaner code)
                    for feature in line:
                        _, value = feature.split(":")
                        exercise_metadata.append(value)
            # Further lines until one exercise is finished represents an instance of this sentence (one word).
            # It contains the word, all of the word-specific features, and the label if the file is train.
            else: 
                # split to word and features
                if key_file:
                    instance_id, word, part_of_speech, morphological_features, dependency_edge_label, dependency_edge_head = line.split()
                else:
                    instance_id, word, part_of_speech, morphological_features, dependency_edge_label, dependency_edge_head, label = line.split()
                    # map this instance's id to the label
                    labels[instance_id] = cast_to_int(label)
                # Sentence Section: extract the word of this instance
                sentence.append(word)
                # Metadata Section
                # concatenate all metadata except morphological_features
                instance_metadata = [part_of_speech, dependency_edge_label, dependency_edge_head]
                # separate morphological features into separate features
                # these features will later become binary categorical features (processing done later, for cleaner code)
                for morph in morphological_features.split("|"):
                    instance_metadata.append(morph)
                # append the result metadata (exercise and instance combined) to metadata matrix
                metadata.append(exercise_metadata + instance_metadata)
                # Label Section:
                instance_ids.append(instance_id)
    if key_file:
        labels = read_key(key_file)
    return sentence_list, metadata_list, instance_ids_list, labels


def create_mappings(metadata_list, u_prev, c_prev):
    """
    This creates mappings for string metadata features that has to be converted to unique ids.
    Parameters:
        metadata_list: a list of metadata for all exercises that were processed from raw file.
    Returns:
        user_to_id: a map of each user_id to a unique integer.
        countries_to_id: a map of each country to a unique integer.
        client_to_id: a map of three kinds of clients: web, android, and ios.
        session_to_id:  a map of three kinds of session: lesson, practice, and test.
        format_to_id: a map of three kinds of format: listen, reverse_tap, and reverse_translate.
        part_of_speech_to_id: maps all unique parts of speech of the target language into a unique integer.
        dependency_edge_label_to_id: maps all unique dependency_edge_labels into a unique integer
        all_morphological_features: a list of all unique morphological features (not a dictionary).
    """
    # create mappings for each metadata feature
    user_to_id, u_next = create_meta_feature_to_id(metadata_list, 0, u_prev)                       #0
    countries_to_id, c_next = create_meta_feature_to_id(metadata_list, 1, c_prev)                  #1
    # days_to_id -> no mapping needed (convert string to float)               #2
    client_to_id = {'web': 0, 'android': 1, 'ios': 2}                         #3
    session_to_id = {'lesson': 0, 'practice': 1, 'test': 2}                   #4
    format_to_id = {'listen': 0, 'reverse_tap': 1, 'reverse_translate': 2}    #5
    # time_to_id -> no mapping needed (convert string to float)               #6
    part_of_speech_to_id = create_feature_to_id(metadata_list, 7)             #7
    dependency_edge_label_to_id = create_feature_to_id(metadata_list, 8)      #8
    # dependency_edge_head_to_id -> no mapping needed (convert string to int) #9
    # morphological_features: convert each feature into a category            #10~
    all_morphological_features = set()
    for metadata in metadata_list:
        for m in metadata:
            for i in range(10, len(m)):
                all_morphological_features.add(m[i])
    all_morphological_features = sorted(list(all_morphological_features))
    return (user_to_id, countries_to_id, client_to_id, session_to_id, format_to_id, part_of_speech_to_id, dependency_edge_label_to_id, all_morphological_features), u_next, c_next


def process_metadata(metadata_list, mappings):
    """
    Using the mappings created above, this converts all of the metadata into numeric values, and reshapes them to a fixed tensor size.
    Parameters:
        metadata_list: the original metadata for all exercises
        mappings: a tuple of all mapping dictionaries and a list of all morphological features.
    Returns:
        a processed python array of the metadata, with the intended shape of (num_of_exercises,(fixed)max_token_size,(fixed)num_of_features)
    """
    user_to_id, countries_to_id, client_to_id, session_to_id, format_to_id, part_of_speech_to_id, dependency_edge_label_to_id, all_morphological_features = mappings
    num_of_features = 10 + len(all_morphological_features)
    for i in range(len(metadata_list)):
        metadata = metadata_list[i]
        for j in range(len(metadata)):
            m = metadata[j]
            # map each feature to its unique id
            m[0] = user_to_id[m[0]] #0 user
            m[1] = countries_to_id[m[1]] #1 countries
            m[2] = cast_to_float(m[2]) #2 days
            m[3] = client_to_id[m[3]] #3 client
            m[4] = session_to_id[m[4]] #4 session
            m[5] = format_to_id[m[5]] #5 format
            m[6] = cast_to_float(m[6]) #6 time
            m[7] = part_of_speech_to_id[m[7]] #7 part_of_speech
            m[8] = dependency_edge_label_to_id[m[8]] #8 dependency_edge_label
            m[9] = cast_to_int(m[9]) #9 dependency_edge_head
            #10 morphological_features
            # create an empty vector of length of all morphological features
            morphological_features = [0] * len(all_morphological_features)
            # for all features in this metadata
            for feature in m[10:]:
                # find index of this feature in the sorted list of all morphological features
                idx = all_morphological_features.index(feature)
                # and map it to 1 to mark that this metadata contains this feature
                morphological_features[idx] = 1
            # update metadata to newly processed attributes
            metadata[j] = m[:10] + morphological_features
        # in order to have a valid input as a tensor, we need to have input size as equal for all inputs
        # we thus pad the metadata matrix with meaningless values (-1) into a shape of (max_token_size, num_of_features)
        dummy = [0] * num_of_features
        metadata += [dummy] * (MAX_TOKEN_SIZE - len(metadata))
        metadata_list[i] = metadata
    return metadata_list


def get_en_es_data(u_prev, c_prev):
    """
    returns en_es train, dev, and test data.
    each processed dataset includes:
        sentences: shape of (num_of_exercises, max_token_size)
        metadata: shape of (num_of_exercises, max_token_size, num_of_features)
        instance_ids: shape of (num_of_exercises, max_token_size)
        labels: a dictionary that maps instance_id to label
    Parameters:
        None
    Return:
        train, dev, and test data.
    """
    print('Loading en_es data')
    # train. 
    print('Loading train dataset')
    sentence_list1, metadata_list1, instance_ids_list1, labels1 = read_data("data_en_es/en_es.slam.20190204.train")
    # create mapping from training data metadata features and use it to process dev and test as well.
    mappings, u_next, c_next = create_mappings(metadata_list1, u_prev, c_prev)
    # convert all items into numpy arrays for tensorflow use
    train_sentence = np.array(sentence_list1) #convert list to np array
    train_metadata = np.array(process_metadata(metadata_list1, mappings))
    train_instance_ids = np.array(instance_ids_list1) 
    en_es_train = (train_sentence, train_metadata, train_instance_ids, labels1)
    
    # dev
    print('Loading dev dataset')
    sentence_list2, metadata_list2, instance_ids_list2, labels2= read_data("data_en_es/en_es.slam.20190204.dev", key_file="data_en_es/en_es.slam.20190204.dev.key")
    #convert list to np array
    dev_sentence = np.array(sentence_list2)
    dev_metadata = np.array(process_metadata(metadata_list2, mappings))
    dev_instance_ids = np.array(instance_ids_list2)
    en_es_dev = (dev_sentence, dev_metadata, dev_instance_ids, labels2)

    #combine train and dev dataset:     
    print('Combining train+dev dataset')
    train_dev_sentence = np.concatenate((train_sentence, dev_sentence), axis=0)
    train_dev_metadata = np.concatenate((train_metadata, dev_metadata), axis=0)
    train_dev_instance_ids = np.concatenate((train_instance_ids, dev_instance_ids), axis=0)

    #shuffle
    index = np.random.permutation(train_dev_sentence.shape[0])
    shuffled_train_dev_sentence = train_dev_sentence[index]
    shuffled_train_dev_metadata = train_dev_metadata[index]
    shuffled_train_dev_instance_ids = train_dev_instance_ids[index]

    #combine two train_labels and dev_labels: no need to shuffle the dict, b/c dict is a set
    combined_labels = copy.deepcopy(labels1)
    combined_labels.update(labels2)

    en_es_train_dev = (shuffled_train_dev_sentence, shuffled_train_dev_metadata, shuffled_train_dev_instance_ids, combined_labels)


    # test
    print('Loading test dataset')
    sentence_list, metadata_list, instance_ids_list, labels = read_data("data_en_es/en_es.slam.20190204.test", key_file="data_en_es/en_es.slam.20190204.test.key")
    en_es_test = (np.array(sentence_list), np.array(process_metadata(metadata_list, mappings)), np.array(instance_ids_list), labels)
    print('Loading complete')

    return en_es_train_dev, en_es_train, en_es_dev, en_es_test, mappings, u_next, c_next
    

def get_es_en_data(u_prev, c_prev):
    """
    returns en_es train, dev, and test data.
    each processed dataset includes:
        sentences: shape of (num_of_exercises, max_token_size)
        metadata: shape of (num_of_exercises, max_token_size, num_of_features)
        instance_ids: shape of (num_of_exercises, max_token_size)
        labels: a dictionary that maps instance_id to label
    Parameters:
        None
    Return:
        train, dev, and test data.
    """
    print('Loading es_en data')
    # train
    print('Loading train dataset')
    sentence_list1, metadata_list1, instance_ids_list1, labels1 = read_data("data_es_en/es_en.slam.20190204.train")
    # es_en_data is a unique case in which test dataset has more morphomological features than the train dataset.
    # In order to create a comprehensive mappings to process the datasets, we manually add the feature that is missing from train.
    (user_to_id, countries_to_id, client_to_id, session_to_id, format_to_id, part_of_speech_to_id, dependency_edge_label_to_id, all_morphological_features), u_next, c_next = create_mappings(metadata_list1, u_prev, c_prev)
    all_morphological_features.append('fPOS=SYM++')
    mappings = (user_to_id, countries_to_id, client_to_id, session_to_id, format_to_id, part_of_speech_to_id, dependency_edge_label_to_id, sorted(all_morphological_features))
    train_sentence = np.array(sentence_list1)
    train_metadata = np.array(process_metadata(metadata_list1, mappings))
    train_instance_ids = np.array(instance_ids_list1)

    es_en_train = (train_sentence, train_metadata, train_instance_ids, labels1)
    
    # dev
    print('Loading dev dataset')
    sentence_list2, metadata_list2, instance_ids_list2, labels2 = read_data("data_es_en/es_en.slam.20190204.dev", key_file="data_es_en/es_en.slam.20190204.dev.key")
    dev_sentence = np.array(sentence_list2)
    dev_metadata = np.array(process_metadata(metadata_list2, mappings))
    dev_instance_ids = np.array(instance_ids_list2)
    es_en_dev =  (dev_sentence, dev_metadata, dev_instance_ids, labels2)

    print('Combining train+dev dataset')
    #combine train + dev
    train_dev_sentence = np.concatenate((train_sentence, dev_sentence), axis=0)
    train_dev_metadata = np.concatenate((train_metadata, dev_metadata), axis=0)
    train_dev_instance_ids = np.concatenate((train_instance_ids, dev_instance_ids), axis=0)

    #shuffle
    index = np.random.permutation(train_dev_sentence.shape[0])
    shuffled_train_dev_sentence = train_dev_sentence[index]
    shuffled_train_dev_metadata = train_dev_metadata[index]
    shuffled_train_dev_instance_ids = train_dev_instance_ids[index]

    #combine two train_labels and dev_labels: no need to shuffle the dict, b/c dict is a set
    combined_labels = copy.deepcopy(labels1)
    combined_labels.update(labels2)

    #when shuffling train and dev

    es_en_train_dev = (shuffled_train_dev_sentence, shuffled_train_dev_metadata, shuffled_train_dev_instance_ids, combined_labels)


    # test
    print('Loading test dataset')
    sentence_list, metadata_list, instance_ids_list, labels = read_data("data_es_en/es_en.slam.20190204.test", key_file="data_es_en/es_en.slam.20190204.test.key")
    es_en_test =  (np.array(sentence_list), np.array(process_metadata(metadata_list, mappings), dtype=np.int32), instance_ids_list, labels)
    print('Loading complete')
    return es_en_train_dev, es_en_train, es_en_dev, es_en_test, mappings, u_next, c_next


def get_fr_en_data(u_prev, c_prev):
    """
    returns en_es train, dev, and test data.
    each processed dataset includes:
        sentences: shape of (num_of_exercises, max_token_size)
        metadata: shape of (num_of_exercises, max_token_size, num_of_features)
        instance_ids: shape of (num_of_exercises, max_token_size)
        labels: a dictionary that maps instance_id to label
    Parameters:
        None
    Return:
        train, dev, and test data.
    """
    print('Loading fr_en data')
    print('Loading train dataset')
    sentence_list1, metadata_list1, instance_ids_list1, labels1 = read_data("data_fr_en/fr_en.slam.20190204.train")
    mappings, u_next, c_next = create_mappings(metadata_list1, u_prev, c_prev)
    train_sentence = np.array(sentence_list1)
    train_metadata = np.array(process_metadata(metadata_list1, mappings))
    train_instance_ids = np.array(instance_ids_list1)
    fr_en_train = (train_sentence, train_metadata, train_instance_ids, labels1)

    print('Loading dev dataset')
    sentence_list2, metadata_list2, instance_ids_list2, labels2 = read_data("data_fr_en/fr_en.slam.20190204.dev", key_file="data_fr_en/fr_en.slam.20190204.dev.key")
    dev_sentence = np.array(sentence_list2)
    dev_metadata = np.array(process_metadata(metadata_list2, mappings))
    dev_instance_ids = np.array(instance_ids_list2)
    fr_en_dev = (dev_sentence, dev_metadata, dev_instance_ids, labels2)

    print('Combining train+dev dataset')
    #combine train+dev
    train_dev_sentence = np.concatenate((train_sentence, dev_sentence), axis=0)
    train_dev_metadata = np.concatenate((train_metadata, dev_metadata), axis=0)
    train_dev_instance_ids = np.concatenate((train_instance_ids, dev_instance_ids), axis=0)

    # shuffle
    index = np.random.permutation(train_dev_sentence.shape[0])
    shuffled_train_dev_sentence = train_dev_sentence[index]
    shuffled_train_dev_metadata = train_dev_metadata[index]
    shuffled_train_dev_instance_ids = train_dev_instance_ids[index]

    # combine two train_labels and dev_labels: no need to shuffle the dict, b/c dict is a set
    combined_labels = copy.deepcopy(labels1)
    combined_labels.update(labels2)

    fr_en_train_dev = (shuffled_train_dev_sentence, shuffled_train_dev_metadata, shuffled_train_dev_instance_ids, combined_labels)


    print('Loading test dataset')
    sentence_list, metadata_list, instance_ids_list, labels = read_data("data_fr_en/fr_en.slam.20190204.test", key_file="data_fr_en/fr_en.slam.20190204.test.key")
    fr_en_test = (np.array(sentence_list), np.array(process_metadata(metadata_list, mappings)), np.array(instance_ids_list), labels)
    print('Loading complete')
    return fr_en_train_dev, fr_en_train, fr_en_dev, fr_en_test, mappings, u_next, c_next
