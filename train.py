import tensorflow as tf 
import numpy as np
from pdb import set_trace as bp
from preprocess import get_es_en_data, get_en_es_data, get_fr_en_data
from IPython import embed
import copy

import eval

from model.encoder_decoder import EncoderDecoder
from model.multi_head_attn import create_padding_mask

from tqdm import tqdm

from utils import PAD_TOKEN, MAX_TOKEN_SIZE, create_token_to_idx, process_sentence

tf.random.set_seed(0)

BATCH_SIZE = 32

def main():

    # load the data
    en_es_train_dev, es_en_train, es_en_dev, es_en_test, mappings1, u1, c1 = get_en_es_data(0,0)

    #for more datasets, uncomment the following two lines
    es_en_train_dev, es_en_train, es_en_dev, es_en_test, mappings2, u2, c2 = get_es_en_data(u1, c1)
    fr_en_train_dev, fr_en_train, fr_en_dev, fr_en_test, mappings3, u3, c3 = get_fr_en_data(u2, c2)

    ##combine train_dev for all three datasets
    #get each attributes
    en_es_sentence, en_es_meta, en_es_inst, en_es_label = en_es_train_dev
    es_en_sentence, es_en_meta, es_en_inst, es_en_label = es_en_train_dev
    fr_en_sentence, fr_en_meta, fr_en_inst, fr_en_label = fr_en_train_dev

    #concatenate
    print(en_es_sentence.shape)
    print(es_en_sentence.shape)
    print(fr_en_sentence.shape)

    print(en_es_meta.shape)
    print(es_en_meta.shape)
    print(fr_en_meta.shape)

    print(en_es_inst.shape)
    print(es_en_inst.shape)
    print(fr_en_inst.shape)
    

    combined_sentence = np.concatenate((en_es_sentence, es_en_sentence, fr_en_sentence), axis=0)
    combined_meta = np.concatenate((en_es_meta, es_en_meta, fr_en_meta), axis=0)
    combined_inst = np.concatenate((en_es_inst, es_en_inst, fr_en_inst), axis=0)
    #combine labels
    combined_labels = copy.deepcopy(en_es_label)
    combined_labels.update(es_en_label) #add es_en to dict
    combined_labels.update(fr_en_label) #add fr_en to dict

    index = np.random.permutation(combined_sentence.shape[0])
    shuffled_combined_sentence = combined_sentence[index]
    shuffled_combined_meta = combined_meta[index]
    shuffled_combined_inst = combined_inst[index]

    combined_train_dev = (shuffled_combined_sentence, shuffled_combined_meta, shuffled_combined_inst, combined_labels)

    #combine mappings1,mappings2,mappings3
    usid1, ctid1, clt1, sessid1, fmatid1, speechid1, dep1, morph1 = mappings1
    usid2, ctid2, clt2, sessid2, fmatid2, speechid2, dep2, morph2 = mappings2
    usid3, ctid3, clt3, sessid3, fmatid3, speechid3, dep3, morph3 = mappings3
    usid = combine_dicts(usid1, usid2, usid3)
    ctid = combine_dicts(ctid, ctid2, ctid3)
    clt = combine_dicts(clt1, clt2, clt3)
    sess = combine_dicts(sessid1, sessid2, sessid3)
    fmat = combine_dicts(fmatid1, fmatid2, fmatid3)
    speech = combine_dicts(speechid1, speechid2, speechid3)
    dep = combine_dicts(dep1, dep2, dep3)
    morph = combine_dicts(morph1, morph2, morph3)
    combined_mappings = (usid, ctid, clt, sess, fmat, speech, dep, morph)
    



    # convert the data to token to idx
    all_tokens = np.array(list(es_en_train[0]) + list(es_en_dev[0]) + list(es_en_test[0]) + \
                          list(en_es_train[0]) + list(en_es_dev[0]) + list(en_es_test[0]) + \
                          list(fr_en_train[0]) + list(fr_en_dev[0]) + list(fr_en_test[0]))
    token_to_idx = create_token_to_idx(all_tokens)

    # TODO: 
    # original code line: train_user_idx = prepare_data(es_en_train[0], es_en_train[1], token_to_idx, user_to_idx)
    # code now: split to processing tokens and importing the already processed metadata
    # the part for metadata originally had a shape of (num_of_exericses, MAX_TOKEN_SIZE) but it's now (num_of_exericses, MAX_TOKEN_SIZE, num_of_features)
    train_sentence_idx = process_sentence(combined_train_dev[0], token_to_idx)
    train_metadata = combined_train_dev[1]

    instance_id_to_dict = combined_train_dev[3]
    # convert true_labels to idx
    labels_array = np.zeros((len(combined_train_dev[0]), MAX_TOKEN_SIZE))
    for i in range(len(labels_array)):
        idx = np.array([instance_id_to_dict[i_id] for i_id in combined_train_dev[2][i]] + \
                       [0] * (MAX_TOKEN_SIZE - len(combined_train_dev[2][i])))
        labels_array[i] = idx

    model = EncoderDecoder(len(token_to_idx), combined_mappings, 300, 300, 4, 100, 100)
    for j in range(10):
        print("Epoch ", j+1)
        # TODO: shuffle training data
        total_loss = 0
        for i in tqdm(range(0, len(train_sentence_idx)/50, BATCH_SIZE)):
            x_batch = train_sentence_idx[i: i+BATCH_SIZE]
            y_batch = labels_array[i: i+BATCH_SIZE]

            x_metadata_batch = train_metadata[i: i+BATCH_SIZE]

            mask = create_padding_mask(x_batch)
            with tf.GradientTape() as tape:
                logits = model.call(x_batch, x_metadata_batch, mask, training=True)
                loss = model.loss_function(logits, y_batch, mask)
                total_loss += loss.numpy()
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print("Avg batch loss", total_loss/(i+1))

            # if i == 40:
            # break
        
        # print("====Dev ====")
        # flattened_instance_ids, actual, preds = predict(model, es_en_dev, token_to_idx)
        print("====Test====")
        flattened_instance_ids1, actual1, preds1 = predict(model, es_en_test, token_to_idx)
        flattened_instance_ids2, actual2, preds2 = predict(model, en_es_test, token_to_idx)
        flattened_instance_ids3, actual3, preds3 = predict(model, fr_en_test, token_to_idx)


def predict(model, data, token_to_idx):
    """The function is used to generate instance-wise predictions;
    The function computes the probability of getting the word 
    incorrect.
    
    Arguments:
        model {tf.model} -- The tensorflow model trained on the duolingo train data
        data {tuple} -- tuple containing the raw_data, user_data, and instance ids
        token_to_idx {dict} -- the mapping from token to idx
        user_to_idx {dict} -- the mapping from user to idx
    """
    raw_sent, raw_users, all_instance_ids, labels_dict  = data
    # TODO: same as above
    sent_idx = process_sentence(raw_sent, token_to_idx)
    user_idx = raw_users
    flattened_instance_ids = []
    
    # create the mask and predict the logits
    mask = create_padding_mask(sent_idx)

    actual = []
    preds = []

    for i in tqdm(range(0, len(sent_idx), BATCH_SIZE)):
        x_batch = sent_idx[i: i+BATCH_SIZE]
        x_user_batch = user_idx[i: i+BATCH_SIZE]
        instance_ids_list = all_instance_ids[i: i+BATCH_SIZE]

        mask = create_padding_mask(x_batch)
        logits = model.call(x_batch, x_user_batch, mask, training=False)
        probs = tf.nn.softmax(logits)
        predictions = probs[:, :, 1]

        # 
        assert len(preds) == len(actual)
        for j, instance_ids in enumerate(instance_ids_list):
            instance_ids_length = len(instance_ids)
            _preds = predictions[j][:instance_ids_length]
            true = [int(labels_dict[instance]) for instance in instance_ids]
            preds.extend(_preds.numpy().tolist())
            actual.extend(true)

            # add to final list of instance ids
            flattened_instance_ids.extend(instance_ids)

    compute_metrics(actual, preds)
    return flattened_instance_ids, actual, preds

def combine_dicts(dict1, dict2, dict3):
    combined_dict = copy.deepcopy(dict1)
    combined_dict.update(dict2)
    combined_dict.update(dict3)
    return combined_dict

def compute_metrics(actual, preds):
    """The function computes all the metrics auroc, f1 and accuracy
    
    Arguments:
        actual {list} -- the list of true labels
        preds {list} -- the list of predictions
    """
    f1 = eval.compute_f1(actual, preds)
    acc = eval.compute_acc(actual, preds)
    auroc = eval.compute_auroc(actual, preds)

    print("AUROC = ", auroc)
    print("F1 = ", f1)
    print("Accuracy = ", acc)
    

def save_preds(filename, flattened_instance_ids, actual, preds):
    pass


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)
    main()
