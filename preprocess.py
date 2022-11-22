import numpy as np
import tensorflow as tf
import pickle
from functools import partial

from utils import get_gt_segments
from baselines_utils import generate_toy_data


AUTOTUNE = tf.data.experimental.AUTOTUNE
MINIGRID_FNAMES = {
    "doorkey": ["DoorKey-8x8-v", 6],
    "keycorridor-s4r3": ["KeyCorridor-S4R3-v", 8],
    "unlockpickup": ["UnlockPickup-v", 10],
    "blockedunlockpickup": ["BlockedUnlockPickup-v", 11]
}


def prepare_dataset(ds, batch_size, dataset_id, model_type="", num_segments=0,
                    max_segment_len=0, pad_len=0, shuffle_buffer_size=30000):
    """
    This is a small dataset, only load it once, and keep it in memory.
    Use `.cache(filename)` to cache preprocessing work for datasets that don't
    fit in memory.
    :param ds: Tensorflow Dataset object
    :param batch_size: batch size
    :param dataset_id: "craft" or "minigrid" or "strings"
    :param model_type: (str) "ompn", "compile", "transformer" etc.
    :param shuffle_buffer_size: Size of the buffer to use for shuffling
    """
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # pad sequences to fixed length for batch-processing and sample a batch
    if dataset_id == "strings":
        # strings 'EOS' token = 0
        pad_len = num_segments*max_segment_len
        ds = ds.padded_batch(batch_size, padded_shapes=([pad_len], []),
                            padding_values=(0.0, 0), drop_remainder=True)

    elif dataset_id == "craft":
        # craft-env 'DONE' = 5 'PAD' = 6
        pad_token = 6.0 if model_type in ["ompn", "compile"] else 5.0
        ds = ds.padded_batch(batch_size, padded_shapes=([pad_len], [pad_len],
                                                        [pad_len, 1075], [], [],
                                                        [pad_len], []),
                             padding_values=(pad_token, 0.0, 0.0, 0, 0, -1, 0),
                             drop_remainder=True)

    elif dataset_id == "minigrid":
        # minigrid-envs 'DONE' = 6 'PAD' = 7
        pad_token = 7.0 if model_type in ["ompn", "compile"] else 6.0
        ds = ds.padded_batch(batch_size, padded_shapes=([pad_len], [pad_len],
                                                        [pad_len, 147], [], [],
                                                        [pad_len], []),
                             padding_values=(pad_token, 0.0, 0.0, 0, 0, -1, 0),
                             drop_remainder=True)
    # `prefetch` lets the dataset fetch batches in the background while the model is training
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def craft_gen(dataset_id, folder, filename, obs_type="", split="train", model_type=""):
    """A Python generator for gym_psketch craft environment demo dataset.
    https://github.com/Ordered-Memory-RL/ompn_craft. """
    # 'DONE' = 5 : EOS token (action) in craft-envs
    trajs = []
    obs_str = obs_type if obs_type == "full" else ""
    if filename == "makeall":
        fnames = ["makebed" + obs_str + "-v0-" + split, "makeaxe" + obs_str + "-v0-" + split,
                  "makeshears" + obs_str + "-v0-" + split]
        for fname in fnames:
            f = open(folder + fname + '.pkl', "rb")
            temp_trajs = pickle.load(f)
            trajs.extend(temp_trajs)
    else:
        f = open(folder + filename + obs_str + "-v0-" + split + '.pkl', "rb")
        # trajectories of (a_t, r_t, o_t)
        trajs = pickle.load(f)

    for i in range(len(trajs)):
        actions, rewards = trajs[i]['action'], trajs[i]['reward']
        features = trajs[i]['features']
        env_id = trajs[i]['env_id']
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        features = np.asarray(features)
        # remove explicitly added 'DONE' token for SloTTAr
        if model_type == "transformer":
            actions = actions[:-1]
            rewards = rewards[:-1]
            features = features[:-1, :]
            # pre-compute gt_num_segments indicator variable for variable slots case
            gt_segments, num_segs = get_gt_segments(dataset_id, filename, actions)
        else:
            gt_segments, num_segs = get_gt_segments(dataset_id, filename, actions[:-1])
        yield (actions, rewards, features, np.asarray(len(actions)), np.asarray(env_id),
               np.asarray(gt_segments), np.asarray(num_segs))


def minigrid_gen(dataset_id, folder, filename, model_type, synth_boundary=0,
                 split="train"):
    """A Python generator for minigrid datasets."""
    fnames, trajs, env_id = [], [], -1
    if filename in MINIGRID_FNAMES.keys():
        fnames = [MINIGRID_FNAMES[filename][0] + str(synth_boundary) + "-"]
        env_id = MINIGRID_FNAMES[filename][1]
    else:
        raise ValueError("Invalid environment name!!!!")

    for fname in fnames:
        # list of episode trajectories of dict element-type
        data = np.load(folder + fname + split + ".npz", allow_pickle=True)
        temp_trajs = data['arr_0']
        trajs.extend(temp_trajs)

    for i in range(len(trajs)):
        actions = np.asarray(trajs[i]['actions'])
        rewards = np.asarray(trajs[i]['rewards'])
        features = np.asarray(trajs[i]['features'])
        # sequence length without added 'DONE' token
        seq_len = len(actions)
        # pre-compute gt_segments and gt_num_segments
        gt_segments, num_segs = get_gt_segments(dataset_id, filename, actions, features)
        # OMPN/CompILE need 'DONE' token explicitly appended to action sequence
        if model_type in ["ompn", "compile"]:
            rewards = np.append(rewards, (0.0,), axis=0)
            actions = np.append(actions, (6,), axis=0)
            features = np.append(features, np.expand_dims(features[-1, :, :, :], axis=0), axis=0)
            if model_type == "compile":
                seq_len = len(actions)
        # actions = (T, 1) rewards = (T, 1), features = (T, 7, 7, 3), env_id = (1,)
        # gt_segments = (T, 1) num_segs (1,)
        yield (actions, rewards, np.reshape(features, (len(actions), -1)),
               np.asarray(seq_len), np.asarray(env_id), np.asarray(gt_segments),
               np.asarray(num_segs))


def strings_gen(num_symbols, num_segments, max_segment_len):
    """Function to randomly generate concatenated segments
        of integer strings dataset on-the-fly"""
    for i in range(128*50):
        input = generate_toy_data(num_symbols=num_symbols, num_segments=num_segments,
                                max_segment_len=max_segment_len)
        yield (np.array(input), np.array(len(input)))


def get_dataset(batch_size, dataset_id, dataset_dir="", dataset_fname="",
                obs_type="", synth_boundary=0, split="train", num_symbols=5,
                num_segments=3, max_segment_len=3, pad_len=0, model_type=""):

    # use different generator methods depending on dataset_id
    if dataset_id == "strings":
        gen_func = partial(strings_gen, num_symbols,
                           num_segments, max_segment_len)
        ds = tf.data.Dataset.from_generator(gen_func, (tf.float32, tf.int32))
    elif dataset_id == "craft":
        gen_func = partial(craft_gen, dataset_id, dataset_dir, dataset_fname, obs_type,
                           split, model_type)
        ds = tf.data.Dataset.from_generator(gen_func, (tf.float32, tf.float32, tf.float32,
                                                       tf.int32, tf.int32, tf.int32, tf.int32))
    elif dataset_id == "minigrid":
        gen_func = partial(minigrid_gen, dataset_id, dataset_dir, dataset_fname, model_type,
                           synth_boundary, split)
        ds = tf.data.Dataset.from_generator(gen_func, (tf.float32, tf.float32, tf.float32,
                                                       tf.int32, tf.int32, tf.int32, tf.int32))
    else:
        raise ValueError("Unknown dataset_id %s" % dataset_id)
    # shuffle and batch ds
    ds = prepare_dataset(ds, batch_size, dataset_id, model_type,
                         num_segments, max_segment_len, pad_len)
    return ds
