import tensorflow as tf
import baselines_utils


class CompILE(tf.keras.Model):
    """CompILE example implementation.

    Args:
        input_dim: Dictionary size of embeddings.
        hidden_dim: Number of hidden units.
        latent_dim: Dimensionality of latent variables (z).
        max_num_segments: Maximum number of segments to predict.
        temp_b: Gumbel softmax temperature for boundary variables (b).
        temp_z: Temperature for latents (z), only if latent_dist='concrete'.
        latent_dist: Whether to use Gaussian latents ('gaussian') or concrete /
            Gumbel softmax latents ('concrete').
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_segments,
                 obs_input=False, temp_b=1., temp_z=1., latent_dist='gaussian'):
        super(CompILE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_num_segments = max_num_segments
        self.obs_input = obs_input
        self.temp_b = temp_b
        self.temp_z = temp_z
        self.latent_dist = latent_dist
        # observation encoder (if obs input available)
        if self.obs_input:
            self.obs_encoder = tf.keras.layers.Dense(hidden_dim, activation=None)

        self.action_embed = tf.keras.layers.Embedding(input_dim, hidden_dim, mask_zero=False)
        # joint encoder of (obs, actions, env_id)
        self.joint_dense_1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.joint_layernorm = tf.keras.layers.LayerNormalization()

        self.lstm_cell = tf.keras.layers.LSTMCell(hidden_dim)

        # LSTM output heads.
        self.head_z_1 = tf.keras.layers.Dense(units=hidden_dim, activation=None)  # Latents (z).

        if latent_dist == 'gaussian':
            self.head_z_2 = tf.keras.layers.Dense(units=2*latent_dim, activation=None)
        elif latent_dist == 'concrete':
            self.head_z_2 = tf.keras.layers.Dense(units=latent_dim, activation=None)
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

        self.head_b_1 = tf.keras.layers.Dense(units=hidden_dim, activation=None)  # Boundaries (b).
        self.head_b_2 = tf.keras.layers.Dense(units=1, activation=None)

        # Decoder MLP.
        self.decode_1 = tf.keras.layers.Dense(units=hidden_dim, activation="relu")
        self.decode_2 = tf.keras.layers.Dense(units=input_dim, activation=None)

    def masked_encode(self, inputs, mask, training):
        """Run masked RNN encoder on input sequence."""
        hidden = baselines_utils.get_lstm_initial_state(inputs.shape[0], self.hidden_dim)
        outputs = []
        for step in range(inputs.shape[1]):
            output, hidden = self.lstm_cell(inputs[:, step], states=hidden, training=training)
            hidden = (mask[:, step, None] * hidden[0], mask[:, step, None] * hidden[1])  # Apply mask.
            outputs.append(output)

        outputs = tf.transpose(tf.convert_to_tensor(outputs), [1, 0, 2])
        return outputs

    def get_boundaries(self, encodings, segment_id, lengths, training):
        """Get boundaries (b) for a single segment in batch."""
        if segment_id == self.max_num_segments - 1:
            # Last boundary is always placed on last sequence element.
            logits_b = None
            sample_b = tf.one_hot(lengths - 1, depth=encodings.shape[1])
        else:
            hidden = tf.keras.activations.relu(self.head_b_1(encodings))
            logits_b = tf.squeeze(self.head_b_2(hidden), axis=-1)
            # Mask out first position with large neg. value.
            neg_inf = tf.ones([encodings.shape[0], 1]) * baselines_utils.NEG_INF
            # TODO(tkipf): Mask out padded positions with large neg. value.
            logits_b = tf.concat([neg_inf, logits_b[:, 1:]], axis=1)
            if training:
                sample_b = baselines_utils.gumbel_softmax_sample(logits_b, temp=self.temp_b)
            else:
                sample_b_idx = tf.math.argmax(logits_b, axis=1)
                sample_b = baselines_utils.to_one_hot(sample_b_idx, logits_b.shape[1])

        return logits_b, sample_b

    def get_latents(self, encodings, probs_b, training):
        """Read out latents (z) form input encodings for a single segment."""
        readout_mask = probs_b[:, 1:, None]  # Offset readout by 1 to left.
        readout = tf.math.reduce_sum(encodings[:, :-1] * readout_mask, axis=1)
        hidden = tf.keras.activations.relu(self.head_z_1(readout))
        logits_z = self.head_z_2(hidden)

        # Gaussian latents.
        if self.latent_dist == 'gaussian':
            if training:
                mu, log_var = tf.split(logits_z, num_or_size_splits=2, axis=1)
                sample_z = baselines_utils.gaussian_sample(mu, log_var)
            else:
                sample_z = logits_z[:, :self.latent_dim]

        # Concrete / Gumbel softmax latents.
        elif self.latent_dist == 'concrete':
            if training:
                sample_z = baselines_utils.gumbel_softmax_sample(logits_z, temp=self.temp_z)
            else:
                sample_z_idx = tf.math.argmax(logits_z, axis=1)
                sample_z = baselines_utils.to_one_hot(sample_z_idx, logits_z.shape[1])
        else:
            raise ValueError('Invalid argument for `latent_dist`.')
        return logits_z, sample_z

    def decode(self, sample_z, length, obs_embed=None):
        """Decode single time step from latents and repeat over full seq."""
        comb_embed = tf.tile(tf.expand_dims(sample_z, axis=1), [1, length, 1])
        # if observation embeddings available concatenate it
        if obs_embed is not None:
            comb_embed = tf.concat([comb_embed, obs_embed], axis=-1)
        pred = self.decode_2(self.decode_1(comb_embed))
        return pred

    def get_next_masks(self, all_b_samples):
        """Get RNN hidden state masks for next segment."""
        if len(all_b_samples) < self.max_num_segments:
            # Product over cumsums (via log->sum->exp).
            log_cumsums = list(map(lambda x: baselines_utils.log_cumsum(x, dim=1), all_b_samples))
            mask = tf.math.exp(sum(log_cumsums))
            return mask
        else:
            return None

    def call(self, actions, lengths, training, obs=None, padding_mask=None):
        # Embed actions.
        embedding = self.action_embed(actions)
        # MLP-Encoder for observation (if given)
        obs_embedding, env_embedding = None, None
        if self.obs_input:
            # encode observations
            obs_embedding = self.obs_encoder(obs)
            # joint Embedding of (obs, actions)
            act_obs_embedding = tf.concat([obs_embedding, embedding], axis=-1)
            embedding = self.joint_layernorm(self.joint_dense_1(act_obs_embedding))

        # Create initial mask.
        mask = tf.ones((actions.shape[0], actions.shape[1]))

        all_b = {'logits': [], 'samples': []}
        all_z = {'logits': [], 'samples': []}
        all_encs = []
        all_recs = []
        all_masks = []
        for seg_id in range(self.max_num_segments):
            # Get masked LSTM encodings of inputs.
            encodings = self.masked_encode(embedding, mask, training=training)
            all_encs.append(encodings)
            # Get boundaries (b) for current segment.
            logits_b, sample_b = self.get_boundaries(encodings, seg_id, lengths, training=training)
            all_b['logits'].append(logits_b)
            all_b['samples'].append(sample_b)
            # Get latents (z) for current segment.
            logits_z, sample_z = self.get_latents(encodings, sample_b, training=training)
            all_z['logits'].append(logits_z)
            all_z['samples'].append(sample_z)
            # Get masks for next segment.
            mask = self.get_next_masks(all_b['samples'])
            all_masks.append(mask)
            # Decode current segment from latents (z).
            reconstructions = self.decode(sample_z, actions.shape[1],
                                          obs_embedding)
            all_recs.append(reconstructions)
        return all_encs, all_recs, all_masks, all_b, all_z
