import tensorflow as tf
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import baselines_utils

eps = 1e-4

PRIORS = {"makeall": np.array([eps, eps, eps, 1.0 - 4*eps, eps], dtype=np.float32),
          "unlockpickup": np.array([eps, eps, 1.0 - 3*eps, eps], dtype=np.float32),
          "doorkey": np.array([eps, eps, 1.0 - 3*eps, eps], dtype=np.float32),
          "blockedunlockpickup":  np.array([eps, eps, eps, 0.85, 0.15], dtype=np.float32),
          "keycorridor-s3r3": np.array([eps, eps, eps, 0.3128, 0.2065, 0.3987,
                                        0.0552, 0.0245, 0.002, eps], dtype=np.float32),
          "keycorridor-s4r3": np.array([eps, eps, eps, 0.318, 0.284, 0.267,
                                        0.089, 0.035, 0.008, eps], dtype=np.float32),
          "keycorridor-s6r3": np.array([eps, eps, eps, 0.2414, 0.2841, 0.3012,
                                        0.1260, 0.0341, 0.0128, eps], dtype=np.float32)
          }


def compute_geometric(lambdas):
    """helper function to compute p_n = lambda_n \prod_{j=1}^{n-1} (1 - lambda_j)
    i.e. equation 2 from PonderNet paper"""
    cumprod_lambdas = tf.math.cumprod(1 - lambdas, axis=-1, exclusive=True)
    p_n = lambdas*cumprod_lambdas
    return p_n


def kl_div(y_pred, dataset_fname):
    prior = tf.convert_to_tensor(PRIORS[dataset_fname])
    kl_loss = y_pred * (tf.math.log(y_pred) - tf.math.log(prior))
    # avg over minibatch samples but sum over dimensions --> 'batchmean' norm in pytorch
    kl_loss = tf.math.reduce_mean(tf.math.reduce_sum(kl_loss, axis=-1), axis=0)
    return kl_loss


def recon_loss(inputs, preds, padding_mask, args, p_k_norm=None):
    """
    Args:
        inputs: padded input sequences
        preds: predicted sequences from SlotAttnSeqAutoEncoder model
        padding_mask: boolean mask sequence for padding tokens
        args: Argument dict from 'ArgumentParser'
        p_k_norm: p_k values (PonderNet loss)
    """
    loss = 0.
    bsz, seq_len = inputs.shape
    targets = tf.reshape(inputs, [-1])
    # (b, 1, 1, T) -> (b, T) -> (b*T)
    padding_mask = tf.reshape(tf.squeeze(padding_mask), [-1])
    # first switch axes (b, n_slots, T, n_symbols) -> (b, T, n_slots, n_symbols)
    preds = tf.transpose(preds, perm=[0, 2, 1, 3])
    preds = tf.reshape(preds, [-1, preds.shape[-1]])
    targets_probs = baselines_utils.to_one_hot(tf.cast(targets, tf.int32), preds.shape[-1])  # (b*T, n_symbols)
    tiled_targets_probs = tf.tile(targets_probs, [args.num_slots, 1])  # (b*T*n_slots,)
    # (n_slots*b*T, n_symbols) -> (n_slots, b, T, n_symbols)
    tiled_targets_probs = tf.reshape(tiled_targets_probs, (args.num_slots, bsz, seq_len,
                                                           args.num_symbols))
    # (n_slots, b, T, n_symbols) -> (b, T, n_slots, n_symbols)
    tiled_targets_probs = tf.transpose(tiled_targets_probs, perm=[1, 2, 0, 3])
    # (b, T, n_slots, n_symbols) -> (b * T * n_slots, n_symbols)
    tiled_targets_probs = tf.reshape(tiled_targets_probs, (bsz * seq_len *
                                                           args.num_slots, args.num_symbols))
    # tiled_padding_mask = tf.tile(padding_mask, [args.num_slots])  # (b*T*n_slots,)
    rec_loss = tf.nn.softmax_cross_entropy_with_logits(tiled_targets_probs, preds)
    # reshape flattened vector -> (b, T, n_slots) and transpose to (b, n_slots, T)
    rec_loss = tf.reshape(rec_loss, [bsz, seq_len, args.num_slots])
    rec_loss = tf.transpose(rec_loss, perm=[0, 2, 1])
    reshaped_padding_mask = tf.reshape(padding_mask, [bsz, 1, seq_len])
    # first compute padded_loss -> output_shape: (b, n_slots)
    masked_rec_loss = tf.math.reduce_sum(rec_loss * reshaped_padding_mask, axis=-1)
    # gate the loss for each variant with num_slots=k with it's halting prob.
    gated_masked_rec_loss = tf.math.reduce_sum(masked_rec_loss*p_k_norm)
    loss = gated_masked_rec_loss / tf.math.count_nonzero(
        tf.cast(padding_mask, dtype=tf.int64), dtype=tf.float32)
    # KL-loss
    kl_term = kl_div(p_k_norm, args.dataset_fname)
    loss += args.beta*kl_term
    return loss


def get_reconstruction_accuracy(inputs, outputs, padding_mask):
    """Calculate reconstruction accuracy averaged over sequence length."""
    inputs = inputs.numpy()
    padding_mask = padding_mask.numpy()
    rec_seq = tf.cast(tf.math.argmax(outputs, axis=-1), dtype=tf.float32)
    # computing number of matches b/w inputs & output sequences
    # TODO: compute rec_acc over non-padded sequence
    matches = np.sum(
        inputs.flatten() * padding_mask.flatten() ==
        rec_seq.numpy().flatten() * padding_mask.flatten())
    rec_acc = matches / inputs.flatten().shape[0]
    return rec_acc, rec_seq


def get_boundaries(sequence):
    """returns sub-task segment boundaries for a
    sequence of segment ids."""
    boundaries = []
    for i in range(len(sequence)-1):
        prev = sequence[i]
        curr = sequence[i+1]
        if prev is not curr:
            boundaries.append(i)
    return boundaries


def get_f1_score(gt_seg_ids, pred_seg_ids, tolerance=1):
    """Compute F-1 score for sub-task boundary segmentation given
    ground-truth and predicted segment_ids."""

    gt_boundaries = get_boundaries(gt_seg_ids)
    pred_boundaries = get_boundaries(pred_seg_ids)

    num_preds = len(pred_boundaries)
    num_targets = len(gt_boundaries)
    correct_preds = 0
    f1_score = 0.0
    precision = 0.0

    for pred_b in pred_boundaries:
        for gt_b in gt_boundaries:
            if gt_b - tolerance <= pred_b <= gt_b + tolerance:
                correct_preds += 1
                break
    if num_preds > 0:
        precision = correct_preds / num_preds

    correct_targets = 0
    for gt_b in gt_boundaries:
        for pred_b in pred_boundaries:
            if pred_b - tolerance <= gt_b <= pred_b + tolerance:
                correct_targets += 1
                break
    recall = correct_targets / num_targets
    if (precision + recall) > 0:
        f1_score = 2*precision*recall / (precision + recall)
    return f1_score


def get_segment_masks(gt_segments, end_pos):
    """helper function that returns segments masks given end positions of each
    segment.
    Example -- input: [1, 1, 2, 3, 2, 2, 5, 0, 2, 2]
    gt_segments: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    """
    start_idx = 0
    gt_seg_id = 0
    end_pos = end_pos.tolist()
    # add last_idx into end_pos if not there (fix for "doorkey" end_idxs)
    if not ((len(gt_segments) - 1) in end_pos):
        end_pos.append(len(gt_segments) - 1)

    for end_idx in end_pos:
        gt_segments[start_idx:end_idx + 1] = gt_seg_id
        start_idx = end_idx + 1
        gt_seg_id = gt_seg_id + 1
    # number of unique elements in segment_mask array == num_segments
    values = np.unique(np.array(gt_segments), return_counts=False)
    num_segments = values.shape[0]
    return gt_segments, num_segments


def pickup_check(pickup_pos, actions, obs):
    true_pickup_pos = []
    for i in list(pickup_pos[0]):
        # OBJ_IDX -> empty: 1, key: 5, ball/boulder: 6
        if (i == len(actions) - 1) or ((obs[i, 3, 6, 0] == 1)
                                       and (obs[i + 1, 3, 6, 0] in [5, 6])):
            true_pickup_pos.append(i)
    return true_pickup_pos


def toggle_check(toggle_pos, obs):
    true_toggle_pos = []
    for i in list(toggle_pos[0]):
        # OBJ_STATE -> open: 0, closed: 1, locked: 2
        if (obs[i, 3, 5, 2] == 2) and (obs[i + 1, 3, 5, 2] == 0):
            true_toggle_pos.append(i)
    return true_toggle_pos


def drop_check(drop_pos, obs):
    true_drop_pos = []
    for i in list(drop_pos[0]):
        if (obs[i, 3, 6, 0] == 6) and (obs[i+1, 3, 6, 0] == 1):
            true_drop_pos.append(i)
    return true_drop_pos


def get_gt_segments(dataset_id, dataset_fname, actions, obs=None):
    """Converts string of integers into their "ground-truth" segment ids.
    Used for computing F1 & Align Acc. eval metrics (from TACO, OMPN papers).
    """
    gt_segments = np.zeros_like(actions, dtype=np.int32)
    if dataset_id == "craft":
        # 'USE' action marks end of segment
        end_pos = np.where(actions == 4)[0]

    elif dataset_id == "minigrid" and (dataset_fname in ["doorkey",
                                                         "unlockpickup"]):
        # 'pickup' or 'toggle' actions mark end of segment
        pickup_pos = np.where(actions == 3)
        # pickup checker based on state change
        true_pickup_pos = pickup_check(pickup_pos, actions, obs)
        # toggle checker based on state change
        toggle_pos = np.where(actions == 5)
        true_toggle_pos = toggle_check(toggle_pos, obs)
        end_pos = np.sort(np.array(true_pickup_pos + true_toggle_pos))

    elif dataset_id == "minigrid" and (dataset_fname == "blockedunlockpickup"):
        # 'pickup' or 'drop' or 'toggle' actions mark end of segment
        pickup_pos = np.where(actions == 3)
        # pickup checker based on state change
        true_pickup_pos = pickup_check(pickup_pos, actions, obs)
        toggle_pos = np.where(actions == 5)
        # toggle checker based on state change
        true_toggle_pos = toggle_check(toggle_pos, obs)
        # concat all "true" boundary pos
        end_pos = np.sort(np.array(true_pickup_pos + true_toggle_pos))

    elif dataset_id == "minigrid" and (dataset_fname in ["keycorridor-s3r3", "keycorridor-s4r3",
                                                         "keycorridor-s6r3"]):
        # 'pickup' or 'toggle' actions mark end of segment
        pickup_pos = np.where(actions == 3)
        # pickup checker based on state change
        true_pickup_pos = pickup_check(pickup_pos, actions, obs)
        toggle_pos = np.where(actions == 5)
        # toggle checker based on state change
        true_toggle_pos = toggle_check(toggle_pos, obs)
        end_pos = np.sort(np.array(true_pickup_pos + true_toggle_pos))

    else:
        raise ValueError("Invalid dataset_id or env_name!!!!")
    gt_segments, num_segments = get_segment_masks(gt_segments, end_pos)
    return gt_segments, num_segments


def get_ordered_segments(pred_seg_ids):
    """helper function to order (in ascending) the segment ids predicted.
    Used for computing alignment accuracy metric. """
    ordered_seg_ids = np.zeros(pred_seg_ids.shape).astype(int)
    pred_seg_ids = pred_seg_ids.tolist()
    unique_seg_ids = []
    for i in range(len(pred_seg_ids)):
        if pred_seg_ids[i] not in unique_seg_ids:
            unique_seg_ids.append(pred_seg_ids[i])
        ordered_seg_ids[i] = unique_seg_ids.index(pred_seg_ids[i])
    return ordered_seg_ids


def get_fw_bw_access(seq_len, slot_attn_weight, alpha_mask, t_on=0.8):
    """
    Args:
        seq_len: unpadded sequence length of action
        slot_attn_weight: 1-d array of slot-attn weights of a sequence
        alpha_mask: 1-d array of alpha-masks of a sequence
        t_on: threshold value to binarize masks
    """
    alpha_max, alpha_min = 0, 0
    fw_access, bw_access = 0., 0.
    binary_slot_attn_weights = (slot_attn_weight > t_on).astype(float)
    binary_alpha_masks = (alpha_mask > t_on).astype(float)
    alpha_on_idxs = np.where(binary_alpha_masks)[0]
    if not alpha_on_idxs.size == 0:
        # handle cases where last mask extends beyond seq_len
        alpha_max = np.clip(np.amax(alpha_on_idxs), 0, seq_len)
        alpha_min = np.clip(np.amin(alpha_on_idxs), 0, seq_len)
        # compute "forward" access = sum_{k=1}^{K} #(slot_attn_k[alpha_max_k+1:seq_len] > t_on) \
        # / # (alpha_max_k:seq_len)
        if alpha_max <= seq_len:
            fw_access = np.count_nonzero(binary_slot_attn_weights[alpha_max+1:seq_len]) / \
                        (seq_len-alpha_max+1)
        # compute "backward" access = sum_{k=1}^{K} #(slot_attn_k)[0:alpha_min_k-1] > t_on) \
        # / # (0:alpha_min_k)
        if alpha_min > 1:
            bw_access = np.count_nonzero(binary_slot_attn_weights[0:alpha_min-1]) / \
                        (alpha_min-1)
    return fw_access, bw_access


def get_eval_metrics(actions, masks, lengths, obs, gt_seg_ids, num_segs,
                     dataset_id, model_type, num_slots=2,
                     slot_attn_weights=None):
    """Calculates mean Adjusted Rand Index, F1 and Alignment Accuracy
    scores on segmentation results."""
    ari_score, f1_score, align_acc = 0., 0., 0.
    forward_access, backward_access = 0., 0.
    pred_seg_ids, pred_seg_idxs = None, None
    actions = actions.numpy()
    masks = masks.numpy()
    lengths = lengths.numpy()
    gt_seg_ids = gt_seg_ids.numpy()
    num_segs = num_segs.numpy()
    if slot_attn_weights is not None:
        slot_attn_weights = slot_attn_weights.numpy()
    bsz = actions.shape[0]

    # for minigrid envs reshaping obs tensor (b, T, flat_dims) -> (b, T, 7, 7, 3)
    if dataset_id == "minigrid":
        obs = np.reshape(obs.numpy(), (bsz, obs.shape[1], 7, 7, 3))
    if model_type == "ompn":
        # for OMPN masks tensor is actually just p_hat values
        masks = tf.transpose(masks, perm=[2, 0, 1])

    for sample_idx in range(bsz):
        # for strings rec_acc is equal to alignment_acc as outputs tokens are itself segment ids
        if dataset_id == "strings":
            # for strings -> actions are already the segment ids (masks)
            gt_seg_ids = actions[sample_idx, :lengths[sample_idx]]
        # compute predicted segment ids from boundary positions
        if model_type == "ompn":
            # Get prediction sorted
            mask = masks[:lengths[sample_idx], sample_idx, :].numpy()
            mask[0, :-1] = 0
            mask[0, -1] = 1
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            p_vals = tf.range(num_slots+1, dtype=tf.float32)
            avg_p = tf.math.reduce_sum(p_vals * mask, axis=-1)
            # always gets oracle number of segments in trajectory
            values, idxs = tf.math.top_k(-avg_p, k=num_segs[sample_idx])
            idxs = idxs.numpy()
            pred_seg_idxs = idxs[tf.argsort(idxs, axis=-1, direction='ASCENDING').numpy()]
            pred_seg_ids, pred_num_segs = get_segment_masks(np.zeros((lengths[sample_idx])), pred_seg_idxs)
        elif model_type == "compile":
            # supervised (oracle) compile only uses gt_num_segs at test time
            pred_seg_ids = tf.math.argmax(masks[sample_idx, :num_segs[sample_idx],
                                          :lengths[sample_idx]]).numpy()
        else:
            # compute predicted segment ids from alpha-masks
            pred_seg_ids = tf.math.argmax(masks[sample_idx, :, :lengths[sample_idx]]).numpy()

        # compute f1-score only on non-padded sequence
        f1_score += get_f1_score(gt_seg_ids[sample_idx, :lengths[sample_idx]].tolist(),
                                 pred_seg_ids.astype(int).tolist())
        # compute ARI only on non-padded sequence
        ari_score += adjusted_rand_score(gt_seg_ids[sample_idx, :lengths[sample_idx]],
                                         pred_seg_ids.astype(int))
        # re-order pred_seg_ids in ascending to compare with gt-segment ids
        align_acc += np.mean(gt_seg_ids[sample_idx, :lengths[sample_idx]] ==
                             get_ordered_segments(pred_seg_ids), axis=0)
        # compute "forward" and "backward" access metrics for our model
        if model_type == "transformer":
            alpha_masks = np.transpose(np.squeeze(masks), axes=[0, 2, 1])
            avg_slot_fw_access, avg_slot_bw_access = 0., 0.
            for slot_k in range(num_segs[sample_idx]):
                trj_fw_access, trj_bw_access = get_fw_bw_access(lengths[sample_idx],
                                                                slot_attn_weights[sample_idx, :, slot_k],
                                                                alpha_masks[sample_idx, :, slot_k])
                avg_slot_fw_access += trj_fw_access
                avg_slot_bw_access += trj_bw_access
            avg_slot_fw_access /= num_segs[sample_idx]
            avg_slot_bw_access /= num_segs[sample_idx]
            forward_access += avg_slot_fw_access
            backward_access += avg_slot_bw_access
    mean_ari_score = ari_score / bsz
    mean_f1_score = f1_score / bsz
    mean_align_acc = align_acc / bsz
    forward_access = forward_access / bsz
    backward_access = backward_access / bsz
    return mean_ari_score, mean_f1_score, mean_align_acc, forward_access, backward_access
