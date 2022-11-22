"""Utility functions."""

import tensorflow as tf
import numpy as np
from gym_psketch import ENV_EMB, env_list, ID2SKETCHLEN

EPS = 1e-17
NEG_INF = -1e30


class EnvEmbedding(tf.keras.Model):
    def __init__(self, emb_size):
        super(EnvEmbedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(len(env_list), emb_size)
        self.emb_size = emb_size

    def call(self, env_ids):
        return self.embedding(env_ids)


class NoEnvEmbedding(tf.keras.Model):
    def __init__(self, emb_size):
        super(NoEnvEmbedding, self).__init__()
        self.emb_size = emb_size

    def call(self, env_ids):
        zeros = tf.zeros_like(env_ids, dtype=tf.float32)
        # final_shape = list(tf.shape(env_ids)) + [self.emb_size]
        return tf.tile(tf.expand_dims(zeros, axis=1), [1, self.emb_size])


def get_env_encoder(env_arch, emb_size):
    if env_arch == 'emb':
        return EnvEmbedding(emb_size)
    elif env_arch == 'noenv':
        return NoEnvEmbedding(emb_size)
    else:
        raise ValueError


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    return tf.one_hot(indices, depth=max_index, dtype=tf.dtypes.float32)


def gumbel_sample(shape):
    """Sample Gumbel noise."""
    uniform = tf.random.uniform(shape, dtype=tf.dtypes.float32)
    return - tf.math.log(EPS - tf.math.log(uniform + EPS))


def gumbel_softmax_sample(logits, temp=1.):
    """Sample from the Gumbel softmax / concrete distribution."""
    gumbel_noise = gumbel_sample(logits.shape)
    return tf.nn.softmax((logits + gumbel_noise) / temp, axis=-1)


def gaussian_sample(mu, log_var):
    """Sample from Gaussian distribution."""
    gaussian_noise = tf.random.normal(mu.shape)
    return mu + tf.math.exp(log_var * 0.5) * gaussian_noise


def kl_gaussian(mu, log_var):
    """KL divergence between Gaussian posterior and standard normal prior."""
    return -0.5 * tf.math.reduce_sum(1 + log_var - mu**2 - tf.math.exp(log_var), axis=1)


def kl_categorical_uniform(preds):
    """KL divergence between categorical distribution and uniform prior."""
    kl_div = preds * tf.math.log(preds + EPS)  # Constant term omitted.
    return tf.math.reduce_sum(kl_div, axis=1)


def kl_categorical(preds, log_prior):
    """KL divergence between two categorical distributions."""
    kl_div = preds * (tf.math.log(preds + EPS) - log_prior)
    return tf.math.reduce_sum(kl_div, axis=1)


def poisson_categorical_log_prior(length, rate):
    """Categorical prior populated with log probabilities of Poisson dist."""
    rate = tf.convert_to_tensor(rate, dtype=tf.dtypes.float32)
    values = tf.expand_dims(tf.range(1, length + 1, dtype=tf.dtypes.float32), axis=0)
    log_prob_unnormalized = tf.math.lgamma(tf.math.log(rate) * values - rate - (values + 1))
    # TODO(tkipf): Length-sensitive normalization.
    return tf.nn.log_softmax(log_prob_unnormalized, axis=1)  # Normalize.


def log_cumsum(probs, dim=1):
    """Calculate log of inclusive cumsum."""
    return tf.math.log(tf.math.cumsum(probs, axis=dim) + EPS)


def generate_toy_data(num_symbols=5, num_segments=3, max_segment_len=5):
    """Generate toy data sample with repetition of symbols (EOS symbol: 0)."""
    seq = []
    min_segment_len = np.floor(0.75*max_segment_len).astype(int)
    symbols = np.random.choice(np.arange(1, num_symbols), num_segments, replace=False)
    for seg_id in range(num_segments):
        segment_len = np.random.choice(np.arange(min_segment_len, max_segment_len))
        seq += [symbols[seg_id]] * segment_len
    return seq


def get_lstm_initial_state(batch_size, hidden_dim):
    """Get empty (zero) initial states for LSTM."""
    hidden_state = tf.zeros((batch_size, hidden_dim), dtype=tf.dtypes.float32)
    cell_state = tf.zeros((batch_size, hidden_dim), dtype=tf.dtypes.float32)
    return hidden_state, cell_state


def get_segment_probs(all_b_samples, all_masks, segment_id):
    """Get segment probabilities for a particular segment ID."""
    neg_cumsum = 1 - tf.math.cumsum(all_b_samples[segment_id], axis=1)
    if segment_id > 0:
        return neg_cumsum * all_masks[segment_id - 1]
    else:
        return neg_cumsum


def get_losses(inputs, outputs, padding_mask, num_segs, args):
    """Get losses (NLL, KL divergences and neg. ELBO).
    Args:
        inputs: Padded input sequences [bsz x pad_len].
        outputs: CompILE model output tuple.
        num_segs: Ground-truth number of segments in a sequence [bsz x 1]
        args: Argument dict from `ArgumentParser`.
    """
    loss, nll, kl_z, kl_b = 0., 0., 0., 0.
    # generate gt_num_segment indicator vector from num_segs scalar
    seg_idxs = np.tile(np.arange(1, args.num_segments+1), (args.batch_size, 1))
    num_segs = num_segs.numpy()
    gt_num_segs_ind = (seg_idxs <= num_segs[:, None])
    gt_num_segs_ind = tf.convert_to_tensor(gt_num_segs_ind, dtype=tf.float32)

    if args.model_type == "compile":
        targets = tf.reshape(inputs, [-1])
        targets_probs = to_one_hot(tf.cast(targets, dtype=tf.dtypes.int32),
                                   args.num_symbols + 1)  # (b*T, n_symbols+1)
        all_encs, all_recs, all_masks, all_b, all_z = outputs
        input_dim = args.num_symbols+1
        nll = 0.
        kl_z = 0.
        for seg_id in range(args.num_segments):
            seg_prob = get_segment_probs(all_b['samples'], all_masks, seg_id)
            preds = tf.reshape(all_recs[seg_id], [-1, input_dim])
            seg_loss = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(
                targets_probs, preds), [-1, inputs.shape[1]])
            # Ignore EOS token (last sequence element) in loss + gate with gt_num_seg indicator variable
            nll += tf.math.reduce_mean(gt_num_segs_ind[:, seg_id] * tf.math.reduce_sum(
                seg_loss[:, :-1] * seg_prob[:, :-1], axis=1), axis=0)
            # KL divergence on z.
            if args.latent_dist == 'gaussian':
                mu, log_var = tf.split(all_z['logits'][seg_id],
                                       num_or_size_splits=2, axis=1)
                # gate kl_z loss with gt_num_seg indicator variable
                kl_z += tf.math.reduce_mean(gt_num_segs_ind[:, seg_id] *
                                            kl_gaussian(mu, log_var), axis=0)
            elif args.latent_dist == 'concrete':
                kl_z += tf.math.reduce_mean(gt_num_segs_ind[:, seg_id] *
                                            kl_categorical_uniform(tf.nn.softmax(
                                                all_z['logits'][seg_id], axis=-1)), axis=0)
            else:
                raise ValueError('Invalid argument for `latent_dist`.')

        # KL divergence on b (first segment only, ignore first time step).
        # TODO(tkipf): Implement alternative prior on soft segment length.
        probs_b = tf.nn.softmax(all_b['logits'][0], axis=-1)
        log_prior_b = poisson_categorical_log_prior(probs_b.shape[1], args.prior_rate)
        kl_b = args.num_segments * tf.math.reduce_mean(kl_categorical(
            probs_b[:, 1:], log_prior_b[:, 1:]), axis=0)
        loss = nll + args.beta * kl_z + args.beta * kl_b
        return loss, nll, kl_z, kl_b

    elif args.model_type == "ompn":
        targets = tf.reshape(inputs, [-1])
        targets_probs = to_one_hot(tf.cast(targets, dtype=tf.dtypes.int32),
                                   args.num_symbols)  # (b*T, n_symbols)
        outputs = tf.reshape(outputs, [-1, outputs.shape[-1]])
        masked_rec_loss = tf.nn.softmax_cross_entropy_with_logits(targets_probs,
                                                                  outputs)
        rec_loss = tf.math.reduce_sum(
            masked_rec_loss * padding_mask) / tf.math.count_nonzero(
            tf.cast(padding_mask, dtype=tf.int64), dtype=tf.float32)
        return rec_loss


def get_reconstruction_accuracy(inputs, outputs, padding_mask, num_segs, args):
    """Calculate reconstruction accuracy (averaged over sequence length)."""
    rec_acc, rec_seq = 0., []
    num_segs = num_segs.numpy()

    if args.model_type == "compile":
        all_encs, all_recs, all_masks, all_b, all_z = outputs
        batch_size = inputs.shape[0]
        for sample_idx in range(batch_size):
            prev_boundary_pos = 0
            rec_seq_parts = []
            for seg_id in range(num_segs[sample_idx]):
                boundary_pos = tf.math.argmax(all_b['samples'][seg_id], axis=-1)[sample_idx]
                if prev_boundary_pos > boundary_pos:
                    boundary_pos = prev_boundary_pos
                seg_rec_seq = tf.math.argmax(all_recs[seg_id], axis=-1)
                rec_seq_parts.append(seg_rec_seq[sample_idx, prev_boundary_pos:boundary_pos])
                prev_boundary_pos = boundary_pos
            rec_seq.append(tf.concat(rec_seq_parts, axis=0))
            cur_length = rec_seq[sample_idx].shape[0]
            matches = rec_seq[sample_idx][:cur_length] == tf.cast(inputs[sample_idx, :cur_length], dtype=tf.int64)
            rec_acc += tf.math.reduce_mean(tf.cast(matches, dtype=tf.float32))
        rec_acc /= batch_size
    elif args.model_type == "ompn":
        inputs = inputs.numpy()
        padding_mask = padding_mask.numpy()
        rec_seq = tf.cast(tf.math.argmax(outputs, axis=-1), dtype=tf.float32)
        # computing number of matches b/w inputs & output sequences
        matches = np.sum(inputs.flatten() * padding_mask.flatten() ==
                         rec_seq.numpy().flatten() * padding_mask.flatten())
        rec_acc = matches / inputs.flatten().shape[0]
    return rec_acc, rec_seq
