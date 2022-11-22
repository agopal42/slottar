import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
from utils import compute_geometric


def create_padding_mask(seq, dataset_id, model_type):
    mask = None
    if dataset_id == "strings":
        mask = tf.cast(tf.math.equal(seq, 0.0), dtype=tf.float32)  # (batch_size, seq_len)
    elif dataset_id == "craft":
        # 'DONE' = 5 ; 'PAD' = 6;
        mask_token = 6.0 if model_type in ["ompn", "compile"] else 5.0
        mask = tf.cast(tf.math.equal(seq, mask_token), dtype=tf.float32)

    elif dataset_id == "minigrid":
        # 'DONE' = 6;  'PAD' = 7
        mask_token = 7.0 if model_type in ["ompn", "compile"] else 6.0
        mask = tf.cast(tf.math.equal(seq, mask_token), dtype=tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def build_seq(seq_length):
    seq_pos = tf.linspace(0., 1., num=seq_length)
    seq_pos = tf.expand_dims(tf.expand_dims(seq_pos, axis=0), axis=-1)
    seq_pos = tf.cast(seq_pos, dtype=tf.float32)
    return seq_pos


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class SoftPositionEmbed(layers.Layer):
    """Adds soft positional embedding with learnable projection."""
    def __init__(self, hidden_size, seq_length):
        """Builds the soft position embedding layer.
        Args:
          hidden_size: Size of input feature dimension.
          seq_length: integer specifying length of sequence.
        """
        super().__init__()
        self.dense = layers.Dense(hidden_size, use_bias=True)
        self.seq_pos = build_seq(seq_length)

    def call(self, inputs):
        return inputs + self.dense(self.seq_pos)


def compute_kth_mask(end_logits_k, mask_upto_km1):
    """function to generate alpha mask for a slot given
    end position logits."""
    end_dist_k = tf.nn.softmax(end_logits_k, axis=1)
    end_cdf_k = tf.math.cumsum(end_dist_k, axis=1)
    mask_upto_k = 1. - end_cdf_k
    mask_k = mask_upto_k * (1. - mask_upto_km1)
    mask_upto_km1 = mask_upto_k
    return mask_k, mask_upto_km1, end_cdf_k


def generate_masks(end_logits, num_slots, extra, p_k, lambdas, training=True):
    # slot-mask generation at training-time
    if training:
        # init stuff
        masks, halting_probs = [], None
        # normalize to get a valid dist. -> p_k
        halting_probs = p_k / tf.math.reduce_sum(p_k, axis=-1, keepdims=True)
        # generate slot-masks
        for K_max in range(num_slots):
            # init stuff
            end_cdf_k, mask_upto_km1 = 1., 0.
            # temp vars
            masks_K_max, masks_K_max_tensor = [], None
            for slot_k in range(K_max+1):
                # compute k^th mask
                mask_k, mask_upto_km1, end_cdf_k = compute_kth_mask(
                                                    end_logits[:, slot_k, :, :],
                                                    mask_upto_km1)
                masks_K_max.append(mask_k)

            masks_K_max_tensor = tf.stack(masks_K_max, axis=1)
            masks.append(masks_K_max_tensor)

    # slot-mask generation at test-time (sampling from Bernoulli to halt)
    else:
        # init stuff
        masks, mask_k,  halting_probs, halting_probs_list = [], None, None, []
        end_cdf_k, mask_upto_km1 = 1., 0.
        already_halted = tf.cast(tf.zeros([end_logits.shape[0], ]), dtype=tf.bool)
        lambda_norm = lambdas / tf.math.reduce_sum(lambdas, axis=-1, keepdims=True)
        # generate slot-masks
        for slot_k in range(num_slots):
            # Bernoulli halting dist.
            halting_dist_k = tfp.distributions.Bernoulli(probs=lambda_norm[:, slot_k])
            # sample from Bernoulli to check halting condition is reached
            halting_probs_k = halting_dist_k.sample()  # 0=continue 1=halt
            # compute k^th mask
            mask_k, mask_upto_km1, end_cdf_k = compute_kth_mask(end_logits[:, slot_k, :, :],
                                                                mask_upto_km1)
            mask_k = mask_k.numpy()
            # check for which sequences halting criteria not reached
            non_halted_idxs = np.where(np.logical_not(already_halted.numpy()))
            # mask of zeros (default)
            zero_mask_k = np.zeros(end_logits[:, 0, :, :].numpy().shape)
            # update masks for non-halted sequences
            zero_mask_k[non_halted_idxs, :, :] = mask_k[non_halted_idxs, :, :]
            # update masks only if not halted
            masks.append(tf.convert_to_tensor(zero_mask_k, dtype=tf.float32))
            halting_probs_list.append(tf.cast(tf.math.logical_not(already_halted), tf.int32))
            # once halted remains halted
            already_halted = tf.math.logical_or(tf.cast(halting_probs_k, dtype=tf.bool),
                                                already_halted)
        halting_probs = tf.stack(halting_probs_list, axis=1)
    return masks, halting_probs, extra


def broadcast_slots(slots, seq_length):
    """Broadcast slot features to a 1D sequence and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    slots = tf.reshape(slots, [-1, slots.shape[-1]])[:, tf.newaxis, :]
    grid = tf.tile(slots, [1, seq_length, 1])
    # `grid` has shape: [batch_size*num_slots, seq_length, slot_size].
    return grid


def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = tf.reshape(x, [batch_size, -1] + x.shape.as_list()[1:])
    channels, masks = tf.split(unstacked, [num_channels, 1], axis=-1)
    return channels, masks


class SlotAttention(layers.Layer):
    """Slot Attention module."""
    def __init__(self, num_iterations, num_slots, slot_size,
                 mlp_hidden_size, slot_stddev=1.0, seed=1, epsilon=1e-8):
        """Builds the Slot Attention module.
        Args:
          num_iterations: Number of iterations.
          num_slots: Number of slots.
          slot_size: Dimensionality of slot feature vectors.
          mlp_hidden_size: Hidden layer size of MLP.
          epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.slot_stddev = slot_stddev
        self.seed = seed
        self.epsilon = epsilon

        # enc_pos added to inputs -> need to LayerNorm again after Transformer Encoder
        self.norm_inputs = layers.LayerNormalization()
        self.norm_slots = layers.LayerNormalization()
        self.norm_mlp = layers.LayerNormalization()

        # Parameters for Gaussian init (no weight-sharing across slots)
        self.slots_mu = self.add_weight(initializer="he_uniform",
                                        shape=[1, self.num_slots, self.slot_size],
                                        dtype=tf.float32, trainable=True,
                                        name="slots_mu")
        self.slots_log_sigma = self.add_weight(initializer="he_uniform",
                                               shape=[1, self.num_slots, self.slot_size],
                                               dtype=tf.float32, trainable=True,
                                               name="slots_log_sigma")

        # Linear maps for the attention module.
        self.project_q = layers.Dense(self.slot_size, use_bias=False, name="q")
        self.project_k = layers.Dense(self.slot_size, use_bias=False, name="k")
        self.project_v = layers.Dense(self.slot_size, use_bias=False, name="v")

        # Slot update functions.
        self.gru = layers.GRUCell(self.slot_size)
        self.mlp = tf.keras.Sequential([layers.Dense(self.mlp_hidden_size, activation="relu"),
                                        layers.Dense(self.slot_size, activation=None)], name="mlp")

    def call(self, inputs, num_iters, mask=None):
        slots, unorm_attn = None, None
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal(
            [tf.shape(inputs)[0], self.num_slots, self.slot_size],
            stddev=self.slot_stddev, seed=self.seed)

        # Multiple rounds of attention.
        for _ in range(num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            q *= self.slot_size ** -0.5  # Normalization.
            attn_logits = tf.keras.backend.batch_dot(k, q, axes=-1)
            # Masking
            if mask is not None:
                # reshape MHA mask tensor (b, 1, 1, L) -> (b, L, 1)
                mask = tf.expand_dims(tf.squeeze(mask), axis=-1)
                attn_logits += (mask * -1e9)
            # Softmax
            attn = tf.nn.softmax(attn_logits, axis=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            # Weighted mean.
            attn += self.epsilon  # attn : (b, L, n_slots)
            # temp var used to log intermediate slot_attn values
            unorm_attn = attn
            attn /= tf.reduce_sum(attn, axis=1, keepdims=True)
            updates = tf.keras.backend.batch_dot(attn, v, axes=-2)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            # Slot update.
            slots, _ = self.gru(updates, [slots_prev])
            slots += self.mlp(self.norm_mlp(slots))
        return slots, unorm_attn


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
                                         layers.Dense(d_model, activation=None)])  # (batch_size, seq_len, d_model)
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, attn_weights = None, None
        # Self-Attention.
        attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2, attn_weights


class Encoder(layers.Layer):
    """Transformer Encoder module."""
    def __init__(self, num_layers, d_model, num_heads, dff, input_size,
                 max_pos_enc, obs_given=True, rate=0.1):
        """
        Args:
            num_layers: # of Transformer layers in the Encoder module
            d_model:
            num_heads: number of attention heads in MHA layer
            dff: hidden size of point-wise feedforward network
            input_size: dimensionality of inputs (vocab size)
            max_pos_enc: maximum position in the input sequence
            obs_given: flag to indicate whether observations are give as inputs.
            rate: Dropout rate
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.input_size = input_size
        self.obs_given = obs_given
        self.max_pos_enc = max_pos_enc
        self.embed_size = d_model

        # action embedding
        self.act_embedding = layers.Embedding(input_size, self.embed_size,
                                              mask_zero=False,
                                              embeddings_initializer="he_uniform")

        if self.obs_given:
            self.joint_dense = layers.Dense(d_model, activation="relu")
            self.joint_norm = layers.LayerNormalization()

        self.pos_enc = positional_encoding(max_pos_enc, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, x, training, obs=None, mask=None):
        attn_mask, attn_weights, pos_emb = None, None, None
        # action embedding
        x = self.act_embedding(x)  # (batch_size, input_seq_len, d_model)
        # (obs + action) joint embedding
        if self.obs_given:
            x = tf.concat([x, obs], axis=-1)
            x = self.joint_norm(self.joint_dense(x))
        # Positional Encoding
        x += self.pos_enc[:, :x.shape[1], :] / tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)
        for layer in range(self.num_layers):
            x, attn_weights = self.enc_layers[layer](x, training, mask)
        return x, attn_weights  # (batch_size, input_seq_len, d_model)


class SloTTArDecoder(layers.Layer):
    """Decoder module for SloTTAr."""
    def __init__(self, num_layers, d_model, num_heads, dff, input_size,
                 max_pos_enc, obs_given=True, rate=0.1):
        """
        Args:
            num_layers: number of Transformer layers in the Decoder module
            d_model: size of Embedding and Encoder layers
            num_heads: number of attention heads in MHA layer
            dff: hidden size of point-wise feedforward network
            input_size: dimensionality of inputs (vocab size)
            max_pos_enc: maximum position in the input sequence
            obs_given: flag to indicate whether inputs are given as inputs.
            rate: Dropout rate
        """
        super(SloTTArDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.input_size = input_size
        self.max_pos_enc = max_pos_enc
        self.obs_given = obs_given

        if self.obs_given:
            self.joint_dense = layers.Dense(d_model, activation="relu")
            self.joint_norm = layers.LayerNormalization()

        self.pos_enc = positional_encoding(max_pos_enc, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training, obs=None, mask=None):
        attn_mask, attn_weights, pos_emb = None, None, None
        # (obs + action) joint embedding
        if self.obs_given:
            x = tf.concat([x, obs], axis=-1)
            x = self.joint_norm(self.joint_dense(x))

        # Positional Encoding
        x += self.pos_enc[:, :x.shape[1], :] / tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)
        for layer in range(self.num_layers):
            x, attn_weights = self.enc_layers[layer](x, training, mask)
        return x, attn_weights  # (batch_size, input_seq_len, d_model)


class SloTTAr(tf.keras.Model):
    """Slot-Attn based Transformer model for learning modular sequence chunks"""
    def __init__(self, num_layers, d_model, num_heads, dff, input_size, seq_len,
                 num_iters, num_slots, slot_size, slot_stddev, target_size,
                 obs_input=True, rate=0.0, seed=1, name="SloTTAr"):
        """
        Args:
            num_layers: # of layers in the Encoder & Decoder of Transformer
            d_model: size of Embedding and Encoder layers
            num_heads: # of attention heads
            dff: hidden size of point-wise feedforward network
            input_size: dimensionality of inputs
            seq_len: sequence length of inputs
            num_iters: # of iterations for slot-attn
            num_slots: # of slots in slot-attn
            slot_size: dimensionality of slots
            slot_stddev: std-dev of Normal dist. for drawing initial slot samples
            target_size: dimensionality of outputs
            obs_input: flag to indicate whether observations are given as inputs.
            rate: Dropout rate param
        """
        super(SloTTAr, self).__init__(name=name)
        self.d_model = d_model
        self.num_slots = num_slots
        self.eps = 1e-8
        self.seed = seed
        self.target_size = target_size
        self.obs_input = obs_input

        if self.obs_input:
            self.obs_encoder = layers.Dense(d_model, activation=None, name="obs_encoder")
            self.obs_decoder = layers.Dense(d_model, activation=None, name="obs_decoder")

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_size, seq_len, obs_input, rate=rate)
        # positional encoding added to slot_attn inputs
        self.encoder_pos = SoftPositionEmbed(d_model, seq_len)
        self.layer_norm = layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([layers.Dense(slot_size, activation="relu"),
                                        layers.Dense(slot_size, activation=None)], name="feedforward")

        self.slot_attn = SlotAttention(num_iters, num_slots, slot_size,
                                       2*slot_size, slot_stddev, seed)

        self.decoder = SloTTArDecoder(num_layers, d_model, num_heads, dff, target_size,
                                      seq_len, obs_input, rate=rate)

        self.output_layer = layers.Dense(target_size, activation=None,
                                         name="output_linear_layer")

    def call(self, actions, training, num_iters, obs=None, pad_mask=None):
        outputs, extra = {}, {}
        lambdas, gains, enc_obs_features = None, None, None
        dec_obs_features, dec_output = None, None
        outputs['p_k'], outputs['lambdas'] = None, None
        outputs['actions'] = actions  # logging actions

        # observation encoder embed
        if self.obs_input:
            enc_obs_features = self.obs_encoder(obs)
            dec_obs_features = self.obs_decoder(obs)

        # inputs (bsz, seq_len, [action; obs_features]) -> (bsz, seq_len, d_model)
        enc_output, extra['enc_attn_weights'] = self.encoder(actions, training,
                                                             obs=enc_obs_features,
                                                             mask=pad_mask)
        # adding positional encoding after Transformer Layer
        enc_output = self.encoder_pos(enc_output)
        # layernorm & mlp
        enc_output = self.mlp(self.layer_norm(enc_output))
        # apply slot-attn
        slots, extra['slot_attn_weights'] = self.slot_attn(enc_output, num_iters,
                                                           pad_mask)
        outputs['slots'] = slots  # logging

        # compute lambda_k from D-th dim of slot representation
        lambdas = tf.keras.activations.sigmoid(slots[:, :, -1])
        # compute p_k from lambdas
        p_k = compute_geometric(lambdas)
        outputs['lambdas'] = lambdas
        outputs['p_k'] = p_k

        # sequential broadcast decoder
        dec_inputs = broadcast_slots(slots, actions.shape[1])
        dec_pad_mask = tf.tile(pad_mask, [self.num_slots, 1, 1, 1])

        # obs_features as input to decoder
        if self.obs_input:
            dec_obs_features = tf.tile(dec_obs_features, [self.num_slots, 1, 1])

        # decoder: dec_output.shape == (batch_size*num_slots, tar_seq_len, d_model)
        dec_output, extra['dec_attn_weights'] = self.decoder(dec_inputs, training,
                                                             obs=dec_obs_features,
                                                             mask=dec_pad_mask)

        # (batch_size, tar_seq_len, target_size) target_size = num_symbols+1 (+1 for mask)
        pred_actions_end_logits = self.output_layer(dec_output)

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, end_logits = unstack_and_split(pred_actions_end_logits,
                                               batch_size=actions.shape[0],
                                               num_channels=self.target_size-1)

        outputs['recons'] = recons  # logging
        # sequentially (left->right) generate masks from end_logits w/ CDF op
        masks_list, halting_probs, extra = generate_masks(end_logits, self.num_slots,
                                                          extra, p_k, lambdas, training)
        outputs['halting_probs'] = halting_probs
        masked_recons, masked_recons_tensor = [], None
        if training:
            for i in range(self.num_slots):
                masked_recons.append(tf.math.reduce_sum(recons[:, :i+1, :, :] * masks_list[i], axis=1))

            masked_recons_tensor = tf.stack(masked_recons, axis=1)
            outputs['masked_recons'] = masked_recons_tensor
        else:
            masks = tf.stack(masks_list, axis=1)
            outputs['masked_recons'] = tf.math.reduce_sum(recons*masks, axis=1)
            outputs['masks'] = masks
        return outputs, extra

