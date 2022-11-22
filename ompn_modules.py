import numpy as np
import tensorflow as tf
import baselines_utils

"""
This code has been adapted to tensorflow from the author's implementation in 
pytorch. Please see relevant files of original pytorch code below for details.
https://github.com/Ordered-Memory-RL/ompn_craft/blob/master/gym_psketch/bots/omdec.py
https://github.com/Ordered-Memory-RL/ompn_craft/blob/master/gym_psketch/bots/om_utils.py
"""


class Distribution(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, dropout, process='softmax'):
        super(Distribution, self).__init__()

        assert process in ['stickbreaking', 'softmax']

        self.mlp_drop_1 = tf.keras.layers.Dropout(dropout)
        self.mlp_dense_1 = tf.keras.layers.Dense(hidden_size, activation=None)
        self.mlp_lrelu_1 = tf.keras.layers.LeakyReLU()
        self.mlp_drop_2 = tf.keras.layers.Dropout(dropout)
        self.mlp_dense_2 = tf.keras.layers.Dense(1, activation=None)

        self.hidden_size = hidden_size
        self.process_name = process

    def init_p(self, bsz, nslot):
        p = np.zeros((bsz, nslot+1))
        p[:, 1] = 1
        return tf.convert_to_tensor(p, dtype=tf.float32)

    @staticmethod
    def process_stickbreaking(beta):
        beta = tf.reverse(beta, [1])
        y = tf.math.cumprod(1 - beta, axis=-1)
        p = tf.pad(beta, tf.constant([[0, 0], [0, 1]]), constant_values=1) * tf.pad(
            y, tf.constant([[0, 0], [1, 0]]), constant_values=1)
        p = tf.reverse(p, [1])
        return p

    @staticmethod
    def process_softmax(beta, mask):
        nslot = beta.shape[1]
        beta = tf.pad(beta, tf.constant([[0, 0], [1, 0]]), constant_values=0)
        beta_normalized = beta - tf.math.reduce_max(beta, axis=-1)[0][:, None]
        x = tf.math.exp(beta_normalized)
        if mask is not None:
            x = tf.repeat(x[:, None, :], repeats=[-1, nslot, -1])
            x = tf.linalg.band_part(x, 0, -1)
            p_candidates, _ = tf.linalg.normalize(x, ord=1, axis=2)
            p = tf.squeeze(tf.linalg.matmul(mask[:, None, :], p_candidates), axis=1)
        else:
            p, _ = tf.linalg.normalize(x, ord=1, axis=1)
        return p

    def call(self, input, training, mask=None):
        mlp_output = self.mlp_dense_2(self.mlp_drop_2(self.mlp_lrelu_1(
            self.mlp_dense_1(self.mlp_drop_1(input, training))), training))
        beta = tf.squeeze(mlp_output, axis=2)
        if self.process_name == 'stickbreaking':
            beta = tf.keras.activations.sigmoid(beta)
            # if mask is not None:
            #     beta = beta * mask.cumsum(dim=-1)
            return self.process_stickbreaking(beta)
        elif self.process_name == 'softmax':
            beta = beta / tf.math.sqrt(self.hidden_size)
            return self.process_softmax(beta, mask)


class Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout=0.2):
        super(Attention, self).__init__()
        self.value = tf.keras.layers.Dense(hidden_size, activation=None)
        self.key = tf.keras.layers.Dense(hidden_size, activation=None)
        self.query = tf.keras.layers.Dense(hidden_size, activation=None)
        self.activation = tf.keras.layers.LayerNormalization()
        self.gating = tf.keras.Sequential([tf.keras.layers.Dense(hidden_size,
                                                                 activation="relu"),
                                           tf.keras.layers.Dense(1, activation="sigmoid")])

        self.hidden_size = hidden_size
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, input, encoded, training):
        memory, memory_mask = encoded
        batch_size, _, _ = input.shape
        key_count, batch_size, _ = memory.shape
        query = self.query(input)
        key = self.key(memory)
        value = self.value(memory)
        scores = tf.einsum('bnd,btd->bnt', (query, key)) / (self.hidden_size ** 0.5)
        scores = tf.where(tf.math.logical_not(tf.cast(memory_mask[:, None, :],
                                                      dtype=tf.bool)), -float('inf'))
        attn = tf.nn.softmax(scores, axis=-1)
        context = tf.einsum('bnt,btd->bnd', (attn, value))

        g_context = self.gating(tf.concat([input, context], axis=-1))
        output = input + g_context * context

        return output


class ComCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout):
        super(ComCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_hidden_size = 4*hidden_size

        self.input_drop_1 = tf.keras.layers.Dropout(dropout)
        self.input_dense_1 = tf.keras.layers.Dense(self.cell_hidden_size, activation="relu")
        self.input_drop_2 = tf.keras.layers.Dropout(dropout)
        self.input_dense_2 = tf.keras.layers.Dense(hidden_size*4)

        self.gates = tf.keras.layers.LayerNormalization()

        self.activation = tf.keras.layers.LayerNormalization(center=False, scale=False)

        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, vi, hi, obs, training):
        """expression for cell function is from Ordered Memory paper
        https://arxiv.org/abs/1910.13466
        """
        input = tf.concat([vi, hi, obs], axis=-1)

        input_activation = self.input_dense_2(self.input_drop_2(
            self.input_dense_1(self.input_drop_1(input, training)), training))
        g_input, cell = tf.split(input_activation, [self.hidden_size*3,
                                                    self.hidden_size], axis=-1)
        gates = tf.keras.activations.sigmoid(self.gates(g_input))
        vg, hg, cg = tf.split(gates, num_or_size_splits=3, axis=1)
        output = self.activation(vg * vi + hg * hi + cg * cell)
        return output


class DecomCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout, attn=False):
        super(DecomCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_hidden_size = 4*hidden_size

        if attn:
            self.attn = Attention(hidden_size=hidden_size)
        else:
            self.attn = None

        self.input_drop_1 = tf.keras.layers.Dropout(dropout)
        self.input_dense_1 = tf.keras.layers.Dense(self.cell_hidden_size,
                                                   activation="relu")
        self.input_drop_2 = tf.keras.layers.Dropout(dropout)
        self.input_dense_2 = tf.keras.layers.Dense(3*hidden_size, activation=None)

        self.gates = tf.keras.layers.LayerNormalization()

        self.drop = tf.keras.layers.Dropout(dropout)

        self.activation = tf.keras.layers.LayerNormalization(center=False,
                                                             scale=False)

    def call(self, inp_enc, parent, training, context=None):
        inputs = tf.concat([parent, inp_enc], axis=1)
        input_activation = self.input_dense_2(self.input_drop_2(
            self.input_dense_1(self.input_drop_1(inputs, training)), training))

        g_input, cell = tf.split(input_activation, [self.hidden_size * 2,
                                                    self.hidden_size * 1], axis=-1)

        gate, cgate = tf.split(tf.keras.activations.sigmoid(self.gates(g_input)),
                               num_or_size_splits=2, axis=-1)
        child = self.activation(gate * self.drop(parent) + cgate * cell)

        if self.attn is not None and context is not None:
            child = tf.squeeze(self.attn(child, context), axis=1)

        return child


class OMStackBot(tf.keras.Model):
    def __init__(self, action_size, slot_size, env_arch, done_id, nb_slots=3,
                 dropout=0.0, process='stickbreaking'):
        super(OMStackBot, self).__init__()

        self.nb_slots = nb_slots
        self.slot_size = slot_size
        self.layernorm = tf.keras.layers.LayerNormalization(center=False, scale=False)
        self.distribution = Distribution(slot_size * 4, slot_size,
                                         dropout=dropout, process=process)
        self.init_p = self.distribution.init_p(1, nslot=self.nb_slots)

        self.actor = tf.keras.layers.Dense(action_size-1, activation=None)
        self.done_id = done_id
        self.memory_encoder = baselines_utils.get_env_encoder(env_arch, slot_size)
        self.encode_obs = tf.keras.layers.Dense(slot_size, name="encode_obs")

        self.com_cell = [ComCell(hidden_size=slot_size, dropout=dropout)
                         for _ in range(nb_slots)]
        self.decom_cell = [DecomCell(hidden_size=slot_size, dropout=dropout)
                           for _ in range(nb_slots-1)]

    def call(self, obs, env_ids, mems=None, training=False):
        p_action, extra_info = None, None
        output_logits, p_hats = [], []
        task_emb = self.memory_encoder(env_ids)
        obs_inp = self.encode_obs(obs)

        for t in range(obs.shape[1]):
            output, mems, extra_info = self.step(obs_inp[:, t, :],
                                                   task_emb, mems, training)

            p_hats.append(extra_info['p_hat'])
            output = tf.concat([output, task_emb, obs_inp[:, t, :]], axis=-1)
            # Replace done with p_end
            output_logit = self.actor(output)
            # computing the prob for 'DONE' token at the end
            p_action = tf.nn.softmax(output_logit, axis=-1)
            p_end = tf.clip_by_value(extra_info['p_end'], 1e-6, 1 - 1e-6)
            p_action = p_action * (1 - p_end)[:, None]
            p_action = tf.concat([p_action[:, :self.done_id], p_end[:, None],
                                  p_action[:, self.done_id:]], axis=1)
            # converting probs -> logits to compute cross_entropy loss
            output_logits.append(tf.math.log(p_action))
        return tf.stack(output_logits, axis=1), tf.stack(p_hats, axis=1), mems, extra_info

    def step(self, input_enc, task_emb, memory, training):
        prev_m, prev_p = self._unflat_memory(memory)
        bsz, nslot, _ = prev_m.shape
        comb_input = tf.concat([input_enc, task_emb], axis=-1)
        p_hat = tf.tile(self.init_p, [bsz, 1])
        cand_m = prev_m
        not_init_id = tf.squeeze(tf.where(tf.math.not_equal(tf.math.reduce_sum(
            prev_p, axis=-1), tf.constant(0.0))), axis=1)

        if len(not_init_id) > 0:
            cm_list = []
            selected_inp = tf.gather(comb_input, not_init_id)
            selected_prev_m = tf.gather(prev_m, not_init_id)
            h = tf.gather(input_enc, not_init_id)

            for i in range(self.nb_slots - 1, -1, -1):
                h = self.com_cell[i](h, selected_prev_m[:, i, :], selected_inp, training)
                cm_list.append(h)
            selected_cand_m = tf.stack(cm_list[::-1], axis=1)
            cand_m = tf.tensor_scatter_nd_update(cand_m, not_init_id[:, None], selected_cand_m)

            dist_input = tf.concat([tf.tile(selected_inp[:, None, :], [1, nslot, 1]),
                                    selected_prev_m, selected_cand_m], axis=-1)
            p_hat = tf.tensor_scatter_nd_update(p_hat, not_init_id[:, None],  self.distribution(dist_input))

        p_end = p_hat[:, 0]
        p, _ = tf.linalg.normalize(p_hat[:, 1:], axis=1, ord=1)
        cp = tf.math.cumsum(p, axis=1)
        rcp = tf.reverse(tf.math.cumsum(tf.reverse(p, [1]), axis=1), [1])

        chl = tf.zeros_like(cand_m[:, 0])
        chl_list = [chl]
        for i in range(self.nb_slots-1):  # last chl not being used anywhere
            h = rcp[:, i, None] * cand_m[:, i] + (1 - rcp)[:, i, None] * chl
            chl = self.decom_cell[i](comb_input, h, training, context=None)
            chl_list.append(chl)
        chl_array = tf.stack(chl_list, axis=1)

        m = prev_m * (1 - cp)[:, :, None] + cand_m * p[:, :, None] + chl_array * (1 - rcp)[:, :, None]
        output = m[:, -1]
        return output, self._flat_memory(m, p), {'p_hat': p_hat, 'p_end': p_end}

    def _flat_memory(self, mem, p):
        batch_size = mem.shape[0]
        mem_size = self.nb_slots * self.slot_size
        return tf.concat([tf.reshape(mem, [batch_size, mem_size]),
                          tf.reshape(p, [batch_size, self.nb_slots])], axis=1)

    def _unflat_memory(self, memory):
        mem_size = self.nb_slots * self.slot_size
        mem = tf.reshape(memory[:, :mem_size], [-1, self.nb_slots, self.slot_size])
        p = memory[:, mem_size:]
        return mem, p

    def init_memory(self, env_ids):
        batch_size = env_ids.shape[0]
        first_slot = self.layernorm(self.memory_encoder(tf.cast(env_ids, dtype=tf.int64)))
        paddings = tf.constant([[0, 0], [0, self.nb_slots - 1], [0, 0]])
        init_m = tf.pad(first_slot[:, None, :], paddings, constant_values=0.)
        init_p = tf.zeros((batch_size, self.nb_slots))
        return tf.concat([tf.reshape(init_m, [batch_size, -1]), init_p], axis=-1)
