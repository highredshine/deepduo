import tensorflow as tf 
import numpy as np
from pdb import set_trace as bp
from preprocess import get_es_en_data, get_en_es_data, get_fr_en_data
from IPython import embed

import eval

from model.encoder_decoder import EncoderDecoder
from model.multi_head_attn import create_padding_mask

from tqdm import tqdm

from utils import PAD_TOKEN, MAX_TOKEN_SIZE, create_token_to_idx, process_sentence

tf.random.set_seed(0)

BATCH_SIZE = 32

def main():
    # load the data
    en_es_train_dev, en_es_train, en_es_dev, en_es_test, mappings, u1, c1 = get_en_es_data(0,0)

    # convert the data to token to idx
    all_tokens = np.array(list(en_es_train[0]) + list(en_es_dev[0]) + list(en_es_test[0]))
    token_to_idx = create_token_to_idx(all_tokens)

    train_sentence_idx = process_sentence(en_es_train_dev[0], token_to_idx)
    train_metadata = en_es_train_dev[1]

    instance_id_to_dict = en_es_train_dev[3]
    # convert true_labels to idx
    labels_array = np.zeros((len(en_es_train_dev[0]), MAX_TOKEN_SIZE))
    for i in range(len(labels_array)):
        idx = np.array([instance_id_to_dict[i_id] for i_id in en_es_train_dev[2][i]] + \
                       [0] * (MAX_TOKEN_SIZE - len(en_es_train_dev[2][i])))
        labels_array[i] = idx

    model = EncoderDecoder(len(token_to_idx), mappings, 100, 100, 4, 100, 100)
    for j in range(10):
        print("Epoch ", j+1)
        # TODO: shuffle training data
        total_loss = 0
        for i in tqdm(range(0, len(train_sentence_idx), BATCH_SIZE)):
            x_batch = train_sentence_idx[i: i+BATCH_SIZE]
            y_batch = labels_array[i: i+BATCH_SIZE]
            # TODO: check whether the changed input with a different shape (explained above) work for this part
            # original code: x_user_batch = train_user_idx[i: i+BATCH_SIZE]
            x_metadata_batch = train_metadata[i: i+BATCH_SIZE]

            mask = create_padding_mask(x_batch)
            with tf.GradientTape() as tape:
                logits = model.call(x_batch, x_metadata_batch, mask, training=True)
                loss = model.loss_function(logits, y_batch, mask)
                total_loss += loss.numpy()
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print("Avg batch loss", total_loss/i+1)

        print("====Test====")
        flattened_instance_ids, actual, preds = predict(model, en_es_test, token_to_idx)


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
