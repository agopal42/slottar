import argparse
import numpy as np
import os
import random
import string
import tensorflow as tf
import wandb

import preprocess
import utils
import baselines_utils
import compile_modules
import ompn_modules
from slottar_modules import create_padding_mask

parser = argparse.ArgumentParser()
# Training
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs.')
parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='Learning rate.')
parser.add_argument('--max_grad_norm', type=float, default=1.0,
                    help='max. norm for gradient clipping.')
parser.add_argument('--model_type', type=str, choices=["compile", "ompn"],
                    default="compile", help='Baseline model architecture.')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--nb_slots', type=int, default=3,
                    help='number of memory slots in OMPN. (hierarchy levels)')
parser.add_argument('--latent_size', type=int, default=128,
                    help='Dimensionality of latent variables. (CompILE)')
parser.add_argument('--latent_dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--beta', type=float, default=0.1,
                    help='loss coefficient for KL-term both latents (CompILE)')
parser.add_argument('--prior_rate', type=float, default=3.0,
                    help='rate parameter for Poisson distribution (CompILE)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Minibatch size.')
parser.add_argument('--max_patience', type=int, default=7,
                    help='maximum patience for early-stopping.')

# Data
parser.add_argument('--dataset_id', type=str, choices=["strings", "craft", "minigrid"],
                    default="minigrid", help='dataset type ("strings" or "craft" or "minigrid").')
parser.add_argument('--dataset_dir', type=str, default="minigrid_data/",
                    help='parent directory containing dataset files.')
parser.add_argument('--dataset_fname', type=str, default="unlockpickup",
                    help='filename of dataset (used for "craft" and "minigrid".')
parser.add_argument('--obs_type', type=str, choices=["partial", "full"],
                    default="full", help='whether offline dataset is full_obs or partial_obs.')
parser.add_argument('--synth_boundary', type=int, default=0,
                    help='0=real dataset; 1=synthetic dataset with 1 boundary '
                         'token (valid only for minigrid).')
parser.add_argument('--num_symbols', type=int, default=6,
                    help='Number of distinct symbols in data generation.')
parser.add_argument('--num_segments', type=int, default=4,
                    help='Number of segments in data generation.')
parser.add_argument('--max_segment_len', type=int, default=6,
                    help='Max. length allowed for each segment in dataset.')
parser.add_argument('--pad_len', type=int, default=65,
                    help='max length to which sequences are padded.')
# Logging
parser.add_argument('--root_dir', type=str, default="",
                    help='Root directory to save logs, ckpts etc.')
parser.add_argument('--log_interval', type=int, default=10,
                    help='Logging interval.')
parser.add_argument('--wandb_logging', type=int, default=0,
                    help='flag to log results on wandb (1) or not.')
parser.add_argument('--save_logs', type=int, default=1,
                    help='Whether to save model ckpts and logs (True) or not.')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='GPU device id to run process on.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed.')

args = parser.parse_args()

PADLEN_NSEGS = {
    "craft": [65, 4],
    "unlockpickup": [35, 3],
    "doorkey": [35, 3],
    "blockedunlockpickup": [40, 5],
    "keycorridor-s4r3": [80, 10]
}


def build_model(args):
    obs_given, model, done_id = None, None, -1
    if args.dataset_id == "strings":
        obs_given = False
    elif args.dataset_id == "craft":
        args.num_symbols, obs_given, done_id = 6, True, 5
    elif args.dataset_id == "minigrid":
        args.num_symbols, obs_given, done_id = 7, True, 6
    else:
        raise ValueError("Unsupported dataset_id name!!!!")

    if args.dataset_id in PADLEN_NSEGS.keys():
        args.pad_len, args.num_segments = PADLEN_NSEGS[
            args.dataset_id]
    elif args.dataset_fname in PADLEN_NSEGS.keys():
        args.pad_len, args.num_segments = PADLEN_NSEGS[
            args.dataset_fname]
    else:
        raise ValueError("Unsupported environment name!!!!")

    if args.model_type == "compile":
        model = compile_modules.CompILE(input_dim=args.num_symbols+1,  # +1 for EOS/Padding symbol.
                                        hidden_dim=args.hidden_size,
                                        latent_dim=args.hidden_size,
                                        max_num_segments=args.num_segments,
                                        latent_dist=args.latent_dist,
                                        obs_input=obs_given)
    elif args.model_type == "ompn":
        model = ompn_modules.OMStackBot(args.num_symbols, args.hidden_size,
                                        "noenv", done_id, args.nb_slots)
    else:
        raise ValueError("Model architecture invalid!!")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    return model, optimizer


def train_step(actions, obs, lengths, env_id, num_segs, model, optimizer):
    extra_info, outputs = {}, None
    loss, nll, kl_z, kl_b = 0., -1., -1., -1.

    # Run forward pass.
    with tf.GradientTape() as tape:
        # create mask tensor
        pad_mask = create_padding_mask(actions, args.dataset_id, args.model_type)
        loss_mask = tf.cast(tf.math.logical_not(tf.cast(pad_mask, dtype=tf.bool)), dtype=tf.float32)
        loss_mask = tf.reshape(tf.squeeze(loss_mask), [-1])
        if args.model_type == "compile":
            outputs = model(actions, lengths, True, obs, env_id)
            loss, nll, kl_z, kl_b = baselines_utils.get_losses(actions, outputs, loss_mask,
                                                               num_segs, args)
        elif args.model_type == "ompn":
            mems = model.init_memory(env_id)
            outputs, p_hats, mems, extra_info = model(obs, env_id, mems, training=True)
            loss = baselines_utils.get_losses(actions, outputs, loss_mask, num_segs, args)
    # update params
    grads = tape.gradient(loss, model.trainable_variables)
    if args.model_type == "ompn":
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=args.max_grad_norm)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, nll, kl_z, kl_b, extra_info


def eval_step(actions, obs, lengths, env_id, gt_segments, num_segs, model, args):
    ari_score, f1_score, align_acc = 0., 0., 0.
    outputs, seg_probs, extra_info = None, None, {}
    # create mask tensor
    pad_mask = create_padding_mask(actions, args.dataset_id, args.model_type)
    # masking tensor for loss/rec_acc computation
    loss_mask = tf.cast(tf.math.logical_not(tf.cast(pad_mask, dtype=tf.bool)),
                        dtype=tf.float32)
    loss_mask = tf.reshape(tf.squeeze(loss_mask), [-1])

    if args.model_type == "compile":
        outputs = model(actions, lengths, False, obs, env_id)
        # compute segment probabilities -> eval_metrics (ARI, F1, Align-Acc.)
        all_encs, all_recs, all_masks, all_b, all_z = outputs
        seg_probs = []
        for seg_idx in range(model.max_num_segments):
            seg_probs.append(baselines_utils.get_segment_probs(all_b['samples'],
                                                               all_masks, seg_idx))
        seg_probs = tf.stack(seg_probs, axis=1)
    elif args.model_type == "ompn":
        mems = model.init_memory(env_id)
        outputs, p_hats, mems, extra_info = model(obs, env_id, mems, training=False)
        seg_probs = tf.transpose(p_hats, perm=[0, 2, 1])
    rec_acc, rec = baselines_utils.get_reconstruction_accuracy(actions, outputs,
                                                               loss_mask, num_segs, args)
    ari_score, f1_score, align_acc, _, _ = utils.get_eval_metrics(actions, seg_probs,
                                                                  lengths, obs, gt_segments,
                                                                  num_segs, args.dataset_id,
                                                                  args.model_type, args.nb_slots)
    return rec_acc, rec, seg_probs, ari_score, f1_score, align_acc, extra_info


def main(args):
    # set random seed.
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    # init system, logging and wandb stuff
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[args.gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu_id], True)

    project_name = args.dataset_id + "-skills"
    if args.wandb_logging:
        wandb.init()  # settings=wandb.Settings(start_method="fork")
        wandb.config.update(args)

    # create dirs for logging/ckpting (if we want to save logs)
    logs_dir, ckpt_prefix = "", ""
    if bool(args.save_logs):
        if args.wandb_logging:
            save_base_dir = os.path.join(args.root_dir, project_name,
                                         wandb.run.name + "-" + wandb.run.id)
        else:
            save_base_dir = os.path.join(project_name, str(''.join(random.choices(
                string.ascii_lowercase, k=5))))

        os.makedirs(save_base_dir)
        logs_dir = os.path.join(save_base_dir, "logs")
        os.makedirs(logs_dir)
        ckpt_prefix = os.path.join(save_base_dir, "ckpt")

    # Build model and init optimizer
    model, optimizer = build_model(args)
    dataset_dir = os.path.join(args.root_dir, args.dataset_id + "_data/")
    # Setup model ckpt
    model_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # Get dataset.
    train_ds = preprocess.get_dataset(args.batch_size, args.dataset_id,
                                      dataset_dir, args.dataset_fname,
                                      args.obs_type, args.synth_boundary,
                                      "train", args.num_symbols,
                                      args.num_segments, args.max_segment_len,
                                      args.pad_len, args.model_type)

    val_ds = preprocess.get_dataset(args.batch_size, args.dataset_id,
                                    dataset_dir, args.dataset_fname,
                                    args.obs_type, args.synth_boundary,
                                    "valid", args.num_symbols,
                                    args.num_segments, args.max_segment_len,
                                    args.pad_len, args.model_type)

    test_ds = preprocess.get_dataset(args.batch_size, args.dataset_id,
                                     dataset_dir, args.dataset_fname,
                                     args.obs_type, args.synth_boundary,
                                     "test", args.num_symbols,
                                     args.num_segments, args.max_segment_len,
                                     args.pad_len, args.model_type)

    # TRAINING LOOP
    print('Training model...')
    actions, rewards, obs, lengths, env_id,  = None, None, None, None, None
    gt_segments, num_segs = None, None
    extra_info = None
    epoch, step = 0, 0
    # val/test vars
    best_val_metrics = -1.0
    # early-stopping patience counter
    patience = 0

    for epoch in range(args.epochs):
        rec, seg_probs = None, None
        for train_batch in train_ds:
            if args.dataset_id == "strings":
                actions, lengths = train_batch
            # craft & minigrid envs also have observation & rewards
            elif args.dataset_id != "strings":
                actions, rewards, obs, lengths, env_id, gt_segments, num_segs = train_batch
            loss, nll, kl_z, kl_b, extra_info = train_step(actions, obs, lengths,
                                                           env_id, num_segs, model,
                                                           optimizer)

            if step % args.log_interval == 0:
                # Run eval.
                rec_acc, rec, seg_probs, ari_score, f1_score, align_acc, \
                extra_info = eval_step(actions, obs, lengths, env_id, gt_segments,
                                       num_segs, model, args)

                # either log stuff to wandb
                if args.wandb_logging:
                    wandb.log({"loss": loss, "nll": nll,
                               "kl_z": kl_z, "kl_b": kl_b,
                               "step": step})
                # or print outputs to console
                else:
                    # ~~~~~~~~~~~~~~~~ print stuff on terminal ~~~~~~~~~~~~~~~~~~~~~
                    print('step: {}, loss: {:.5f}, rec_acc: {:.4f} '
                          'ari_score: {:.4f} f1_score: {:.4f} align_acc: {:.4f}'
                          .format(step, loss, rec_acc, ari_score, f1_score, align_acc))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # update step
            step = step + 1

        # VALID/TEST EVAL
        val_actions, val_rewards, val_obs, val_lengths, val_env_id = None, None, None, None, None,
        val_gt_segments, val_num_segs = None, None
        val_rec_acc, val_ari_score, val_f1_score, val_align_acc, val_steps = 0., 0., 0., 0., 0
        test_actions, test_rewards, test_obs, test_lengths, test_env_id = None, None, None, None, None
        test_gt_segments, test_num_segs = None, None
        test_rec_acc, test_ari_score, test_f1_score, test_align_acc, test_steps = 0., 0., 0., 0., 0

        for val_batch, test_batch in zip(val_ds, test_ds):
            if args.dataset_id == "strings":
                val_actions, val_lengths = val_batch
                test_actions, test_lengths = test_batch
            # craft & minigrid envs also have observation & rewards
            elif args.dataset_id != "strings":
                val_actions, val_rewards, val_obs, val_lengths, val_env_id, val_gt_segments, val_num_segs = val_batch

                test_actions, test_rewards, test_obs, test_lengths, test_env_id, test_gt_segments, test_num_segs = test_batch

            # Run evaluation on validation set.
            rec_acc, rec, seg_probs, ari_score, f1_score, align_acc, \
            extra_info = eval_step(val_actions, val_obs, val_lengths, val_env_id,
                                   val_gt_segments, val_num_segs, model, args)
            val_rec_acc += rec_acc
            val_ari_score += ari_score
            val_f1_score += f1_score
            val_align_acc += align_acc
            val_steps += 1

            # Run evaluation on test set.
            rec_acc, rec, seg_probs, ari_score, f1_score, align_acc, \
            extra_info = eval_step(test_actions, test_obs, test_lengths, test_env_id,
                                   test_gt_segments, test_num_segs, model, args)

            test_rec_acc += rec_acc
            test_ari_score += ari_score
            test_f1_score += f1_score
            test_align_acc += align_acc
            test_steps += 1

        # epoch done
        epoch += 1

        # log eval metrics on val/test splits to wandb or print them out
        if args.wandb_logging:
            # log validation set eval_metrics to wandb
            wandb.log({"val_rec_acc": val_rec_acc/val_steps,
                       "val_ari_score": val_ari_score/val_steps,
                       "val_f1_score": val_f1_score/val_steps,
                       "val_align_acc": val_align_acc/val_steps,
                       "epoch": epoch})

            # log test set eval_metrics to wandb
            wandb.log({"test_rec_acc": test_rec_acc/test_steps,
                       "test_ari_score": test_ari_score/test_steps,
                       "test_f1_score": test_f1_score/test_steps,
                       "test_align_acc": test_align_acc/test_steps,
                       "epoch": epoch})
        else:
            # ~~~~~ print validation/test stuff to terminal ~~~~~~
            print()
            print('epoch: {} val_rec_acc: {:.4f} val_ari_score {:.4f} '
            'val_f1_score {:.4f} val_align_acc {:.4f}'.format(epoch, val_rec_acc/val_steps,
            val_ari_score/val_steps, val_f1_score/val_steps, val_align_acc/val_steps))
            print('epoch: {} test_rec_acc: {:.4f} test_ari_score {:.4f} '
            'test_f1_score {:.4f} test_align_acc {:.4f}'.format(epoch, test_rec_acc/test_steps,
            test_ari_score/test_steps, test_f1_score/test_steps, test_align_acc/test_steps))
            print()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        avg_val_metrics = np.mean([val_ari_score / val_steps, val_f1_score / val_steps,
             val_align_acc / val_steps])
        # ckpt if avg_val_metrics have improved
        if avg_val_metrics > best_val_metrics:
            best_val_metrics = avg_val_metrics
            # reset patience for early-stopping
            patience = 0
            # log best test set eval_metrics
            if args.wandb_logging:
                wandb.log({"best_ari_score": test_ari_score / test_steps,
                           "best_f1_score": test_f1_score / test_steps,
                           "best_align_acc": test_align_acc / test_steps,
                           "epoch": epoch})
            if args.save_logs:
                # ckpt-model if avg. of all val_metrics improve
                model_ckpt.save(ckpt_prefix)
                # saving data for viz if avg. of all val_metrics improve end of epoch
                np.savez(logs_dir + "/" + "eval_logs_" + str(epoch) + ".npz",
                          actions=actions.numpy(), rec=np.array(rec, dtype="object"),
                         seg_probs=seg_probs.numpy(), lengths=lengths.numpy())
        # early-stopping counter update
        else:
            patience += 1
            print('early-stopping patience count: {}'.format(patience))
        # stop training when max_patience is reached
        if patience == args.max_patience:
            break

    return 0


if __name__ == "__main__":
    main(args)
