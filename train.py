import argparse
import string
import random
import tensorflow as tf
import numpy as np
import wandb
import os

import utils
from slottar_modules import SloTTAr, create_padding_mask
import preprocess

parser = argparse.ArgumentParser()
# Training
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training iterations.')
parser.add_argument('--max_patience', type=int, default=7,
                    help='Maximum patience for early-stopping.')
parser.add_argument('--learning_rate', type=float, default=0.0004,
                    help='Learning rate.')
parser.add_argument('--max_norm', type=int, default=1.0,
                    help='global norm to clip gradients.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Mini-batch size.')
parser.add_argument('--beta', type=float, default=1.0,
                    help='beta weight for KL term.')
# Architecture
parser.add_argument('--model_type', type=str, default="transformer",
                    help='Type of model architecture.')
parser.add_argument('--hidden_size', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--num_layers', type=int, default=1,
                    help='Number of layers in Transformer model.')
parser.add_argument('--num_heads', type=int, default=4,
                    help='Number of attn-heads')
parser.add_argument('--slot_size', type=int, default=32,
                    help='Dimensionality of slot representation.')
parser.add_argument('--slot_stddev', type=float, default=1.0,
                    help='std-dev of Gaussian to draw noise samples.')
parser.add_argument('--num_slots', type=int, default=4,
                    help='Number of slots in slot-attn module.')
parser.add_argument('--num_iters', type=int, default=1,
                    help='Number of iterations used in slot-attn.')
# Data
parser.add_argument('--dataset_id', type=str, choices=["strings", "craft", "minigrid"],
                    default="craft", help='dataset name.')
parser.add_argument('--dataset_fname', type=str, default="",
                    help='filename of dataset (used for "craft" and "minigrid".')
parser.add_argument('--obs_type', type=str, default="", choices=["full", "partial"],
                    help='whether offline dataset is full_obs or partial_obs.')
parser.add_argument('--synth_boundary', type=int, default=0,
                    help='0=real dataset; 1=synthetic dataset with 1 boundary '
                         'token (valid only for minigrid).')
parser.add_argument('--num_symbols', type=int, default=6,
                    help='Number of symbols in data. (strings=6, craft=6, minigrid=7')
parser.add_argument('--num_segments', type=int, default=4,
                    help='Number of segments in data generation.')
parser.add_argument('--max_segment_len', type=int, default=6,
                    help='Max. length allowed for each segment in dataset.')
parser.add_argument('--pad_len', type=int, default=65,
                    help='max length to which sequences are padded.')
# Eval/Logging
parser.add_argument('--root_dir', type=str, default="",
                    help='Root directory to save logs, ckpts, load data etc.')
parser.add_argument('--log_interval', type=int, default=10,
                    help='Logging interval.')
parser.add_argument('--save_logs', type=int, default=0,
                    help='Whether to save model ckpts and logs (1) or not (0).')
parser.add_argument('--wandb_logging', type=int, default=0,
                    help='flag to log results on wandb (1) or not.')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='GPU device id to run process on.')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed.')

args = parser.parse_args()

PADLEN_NSEGS_NSLOTS = {
    "craft": [65, 4, 5],
    "unlockpickup": [35, 3, 4],
    "doorkey": [35, 3, 4],
    "blockedunlockpickup": [40, 5, 5],
    "keycorridor-s4r3": [80, 10, 10]
}


def build_model(args):
    obs_given, model = None, None
    # init optimizers
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,
                                         epsilon=1e-8)
    # inferring some dataset-specific hyperparams
    if args.dataset_id == "strings":
        obs_given = False
    elif args.dataset_id == "craft":
        obs_given, args.num_symbols = True, 6
    elif args.dataset_id == "minigrid":
        obs_given, args.num_symbols = True, 7
    else:
        raise ValueError("Unsupported dataset_id name!!!!")

    if args.dataset_id in PADLEN_NSEGS_NSLOTS.keys():
        args.pad_len, args.num_segments, args.num_slots = PADLEN_NSEGS_NSLOTS[
            args.dataset_id]
    elif args.dataset_fname in PADLEN_NSEGS_NSLOTS.keys():
        args.pad_len, args.num_segments, args.num_slots = PADLEN_NSEGS_NSLOTS[
            args.dataset_fname]
    else:
        raise ValueError("Unsupported environment name!!!!")

    if args.model_type == "transformer":
        model = SloTTAr(args.num_layers, args.hidden_size, args.num_heads,
                        4*args.hidden_size, args.num_symbols, args.pad_len,
                        args.num_iters, args.num_slots, args.slot_size, args.slot_stddev,
                        args.num_symbols+1, obs_input=obs_given, seed=args.seed)
    return model, optimizer


def train_step(actions, obs, model, optimizer, args):
    # Run forward pass.
    with tf.GradientTape() as tape:
        # create mask tensor
        pad_mask = create_padding_mask(actions, args.dataset_id, args.model_type)
        outputs, extra = model(actions, training=True, num_iters=args.num_iters,
                               obs=obs, pad_mask=pad_mask)
        loss_mask = tf.cast(tf.math.logical_not(tf.cast(pad_mask, dtype=tf.bool)),
                            dtype=tf.float32)
        loss = utils.recon_loss(actions, outputs['masked_recons'], loss_mask,
                                args, outputs['halting_probs'])
    # Update params
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return outputs, extra, loss


def eval_step(actions, obs, lengths, gt_segments, num_segs, model, args):
    # create mask tensor
    pad_mask = create_padding_mask(actions, args.dataset_id, args.model_type)
    outputs, extra = model(actions, training=False, num_iters=args.num_iters,
                           obs=obs, pad_mask=pad_mask)
    masks = tf.squeeze(outputs['masks'], axis=-1)
    # masking tensor for loss/rec_acc computation
    loss_mask = tf.cast(tf.math.logical_not(tf.cast(pad_mask, dtype=tf.bool)),
                        dtype=tf.float32)
    loss_mask = tf.reshape(tf.squeeze(loss_mask), [-1])
    rec_acc, rec = utils.get_reconstruction_accuracy(actions, outputs['masked_recons'], loss_mask)
    ari_score, f1_score, align_acc, \
    fw_access, bw_access = utils.get_eval_metrics(actions, masks, lengths,
                                                  obs, gt_segments, num_segs,
                                                  args.dataset_id, args.model_type,
                                                  args.num_slots, extra['slot_attn_weights'])
    return outputs, extra, rec_acc, rec, ari_score, f1_score, \
           align_acc, fw_access, bw_access


def main(args):

    # set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # init system, logging and wandb stuff
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[args.gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu_id], True)

    project_name = args.dataset_id + "-skills"
    if args.wandb_logging:
        wandb.init()
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

    # Setup model ckpt
    model_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    dataset_dir = os.path.join(args.root_dir, args.dataset_id + "_data/")
    # Get dataset.
    train_ds = preprocess.get_dataset(args.batch_size, args.dataset_id,
                                      dataset_dir, args.dataset_fname,
                                      args.obs_type, args.synth_boundary,
                                      "train", args.num_symbols, args.num_segments,
                                      args.max_segment_len, args.pad_len,
                                      args.model_type)

    val_ds = preprocess.get_dataset(args.batch_size, args.dataset_id,
                                    dataset_dir, args.dataset_fname,
                                    args.obs_type, args.synth_boundary,
                                    "valid", args.num_symbols, args.num_segments,
                                    args.max_segment_len, args.pad_len,
                                    args.model_type)

    test_ds = preprocess.get_dataset(args.batch_size, args.dataset_id,
                                     dataset_dir, args.dataset_fname,
                                     args.obs_type, args.synth_boundary,
                                     "test", args.num_symbols,  args.num_segments,
                                     args.max_segment_len, args.pad_len,
                                     args.model_type)

    # TRAINING LOOP
    print('Training model .....')
    epoch, step = 0, 0
    actions, rewards, obs, lengths, env_id = None, None, None, None, None
    gt_segments, num_segs = None, None
    # val/test vars
    best_val_metrics = -1.0

    # early-stopping counter
    patience = 0

    for epoch in range(args.epochs):
        outputs, extra = {}, {}
        for train_batch in train_ds:
            if args.dataset_id == "strings":
                actions, lengths = train_batch
            # craft & minigrid envs also have observation & rewards
            elif args.dataset_id != "strings":
                actions, rewards, obs, lengths, env_id, gt_segments, num_segs = train_batch

            outputs, extra, loss = train_step(actions, obs, model, optimizer, args)

            if step % args.log_interval == 0:
                # Run evaluation.
                outputs, extra, rec_acc, rec, ari_score, f1_score, align_acc, \
                fw_access, bw_access = eval_step(actions, obs, lengths, gt_segments,
                                                 num_segs, model, args)
                # log stuff to wandb
                if args.wandb_logging:
                    wandb.log({"loss": loss.numpy(),
                            "rec_accuracy": rec_acc,
                            "ari_score": ari_score,
                            "f1_score": f1_score,
                            "align_acc": align_acc,
                            "step": step})
                else:
                    # ~~~~~ print out stuff to terminal ~~~~~~
                    print('step: {}, ce_loss_train: {:.5f}, rec_accuracy: {:.5f} '
                          'ari_score: {:.4f}, f1_score: {:.4f}, align_acc: {:.4f}'
                          .format(step, loss.numpy(), rec_acc, ari_score, f1_score, align_acc))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # update step
            step = step + 1
        # end of training epoch

        # VALID/TEST EVAL
        val_actions, val_rewards, val_obs, val_lengths, val_env_id = None, None, None, None, None
        val_gt_segments, val_num_segs = None, None
        val_rec_acc, val_ari_score, val_f1_score, val_align_acc, val_steps = 0., 0., 0., 0., 0
        val_fw_access, val_bw_access = 0., 0.
        test_actions, test_rewards, test_obs, test_lengths, test_env_id = None, None, None, None, None
        test_gt_segments, test_num_segs = None, None
        test_rec_acc, test_ari_score, test_f1_score, test_align_acc, test_steps = 0., 0., 0., 0., 0
        test_fw_access, test_bw_access = 0., 0.

        for val_batch, test_batch in zip(val_ds, test_ds):
            if args.dataset_id == "strings":
                val_actions, val_lengths = val_batch
                test_actions, test_lengths = test_batch
            # craft & minigrid envs also have observation & rewards
            elif args.dataset_id != "strings":
                val_actions, val_rewards, val_obs, val_lengths, val_env_id, \
                val_gt_segments, val_num_segs = val_batch
                test_actions, test_rewards, test_obs, test_lengths, test_env_id, \
                test_gt_segments, test_num_segs = test_batch

            # Run evaluation on validation set.
            outputs, extra, rec_acc, rec, ari_score, f1_score, align_acc, \
            fw_access, bw_access = eval_step(val_actions, val_obs, val_lengths,
                                             val_gt_segments, val_num_segs,
                                             model, args)
            val_rec_acc += rec_acc
            val_ari_score += ari_score
            val_f1_score += f1_score
            val_align_acc += align_acc
            val_fw_access += fw_access
            val_bw_access += bw_access
            val_steps = val_steps + 1

            # Run evaluation on test set.
            outputs, extra, rec_acc, rec, ari_score, f1_score, align_acc, \
            fw_access, bw_access = eval_step(test_actions, test_obs, test_lengths,
                                             test_gt_segments, test_num_segs,
                                             model, args)

            test_rec_acc += rec_acc
            test_ari_score += ari_score
            test_f1_score += f1_score
            test_align_acc += align_acc
            test_fw_access += fw_access
            test_bw_access += bw_access
            test_steps += 1

        # incrementing epoch counter
        epoch += 1

        if args.wandb_logging:
            # log validation set eval_metrics to wandb
            wandb.log({"val_rec_acc": val_rec_acc / val_steps,
                       "val_ari_score": val_ari_score / val_steps,
                       "val_f1_score": val_f1_score / val_steps,
                       "val_align_acc": val_align_acc / val_steps,
                       "val_fw_access": val_fw_access / val_steps,
                       "val_bw_access": val_bw_access / val_steps,
                       "epoch": epoch})
            # log test set eval_metrics to wandb
            wandb.log({"test_rec_acc": test_rec_acc / test_steps,
                       "test_ari_score": test_ari_score / test_steps,
                       "test_f1_score": test_f1_score / test_steps,
                       "test_align_acc": test_align_acc / test_steps,
                       "test_fw_access": test_fw_access / test_steps,
                       "test_bw_access": test_bw_access / test_steps,
                       "epoch": epoch})

        else:
            # ~~~~~ print valid/test set metrics to terminal ~~~~~~
            print()
            print('epoch: {}, val_rec_acc: {:.4f} '
                  'val_ari_score: {:.4f}, val_f1_score: {:.4f}, val_align_acc: {:.4f} '
                  'val_fw_access: {:.4f}, val_bw_access: {:.4f} '
                  .format(epoch, val_rec_acc/val_steps, val_ari_score/val_steps,
                          val_f1_score/val_steps, val_align_acc/val_steps,
                          val_fw_access/val_steps, val_bw_access/val_steps))
            print('epoch: {}, test_rec_acc: {:.4f} '
                  'test_ari_score: {:.4f}, test_f1_score: {:.4f}, test_align_acc: {:.4f} '
                  'test_fw_access: {:.4f}, test_bw_access: {:.4f}'
                  .format(epoch, test_rec_acc/test_steps, test_ari_score/test_steps,
                          test_f1_score/test_steps, test_align_acc/test_steps,
                          test_fw_access/test_steps, test_bw_access/test_steps))
            print()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        avg_val_metrics = np.mean([val_f1_score/val_steps,
                                   val_align_acc/val_steps])

        # ckpt if avg_val_metrics improve
        if avg_val_metrics > best_val_metrics:
            best_val_metrics = avg_val_metrics
            # reset patience for early-stopping
            patience = 0
            # log best test set eval_metrics
            if args.wandb_logging:
                wandb.log({"best_ari_score": test_ari_score/test_steps,
                           "best_f1_score": test_f1_score/test_steps,
                           "best_align_acc": test_align_acc/test_steps,
                           "epoch": epoch})
            if bool(args.save_logs):
                # ckpt-model if avg val_metrics improve
                model_ckpt.save(ckpt_prefix)
                # saving data for viz if avg. of all val_metrics improve end of epoch
                np.savez(logs_dir + "/" + "eval_logs_" + str(epoch) + ".npz",
                         **outputs, **extra)
        # increment early-stopping counter if no improvement
        else:
            patience += 1
            print('early-stopping patience count: {}'.format(patience))
        # stop training when max_patience is reached
        if patience == args.max_patience:
            break
    return 0


if __name__ == "__main__":
    main(args)
