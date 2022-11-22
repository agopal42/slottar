# SloTTAr
This is the code repository complementing the paper. 

**Unsupervised Learning of Temporal Abstractions using Slot-based Transformers**  
Anand Gopalakrishnan, Kazuki Irie, J&uuml;rgen Schmidhuber, Sjoerd van Steenkiste
https://arxiv.org/abs/2203.13573

## Dependencies:
Requires Python 3.6 or later. Please install all the dependencies listed in the `requirements.txt` file. 
Additionally, for the experiments using the `Craft` suite of environments please install the `gym_psketch` library 
by following the instructions here (https://github.com/Ordered-Memory-RL/ompn_craft).  

## Dataset:
The offline Craft datasets of rollouts used in this paper can be downloaded [here](https://drive.google.com/drive/folders/11YdDK8omZ4O5CR7rOQQLYVwklwYkMnV1?usp=share_link).
The offline MiniGrid datasets of rollouts used in this paper can be downloaded [here](https://drive.google.com/drive/folders/1RqeRNllVvg6mxjfGBGyphikJzaXlRehx?usp=share_link).
You can pass the parent directory under which these dataset folders are stored to training scripts
using the flag `root_dir`.

## Files:
* `baseline_utils.py`: utility functions used by the baseline models (CompILE & OMPN) .
* `compile_modules.py`: all the neural network modules for CompILE baseline model.
* `ompn_modules.py`: all the neural network modules for OMPN baseline model. 
* `preprocess.py`: data loader and utility functions for preprocessing the offline datasets.
* `slottar_modules.py`: all the neural network modules for SloTTAr (our model).
* `train.py`: training script for SloTTAr (our model). 
* `train_baselines.py`: training script for baseline models (CompILE & OMPN).
* `viz.ipynb`: jupyter-notebook with helper functions and commands for all visualizations/plots in the paper.

## Experiments:
Following are some example commands to recreate some experimental results in the paper.

For the SloTTAr results in Table 1. on `Craft (fully)` initialize and run the wandb sweep:
```commandline
wandb sweep exp_configs/slottar_craftf.yaml
wandb agent SWEEP_ID
```
For the SloTTAr results in Table 1. on `Craft (partial)` initialize and run the wandb sweep:
```commandline
wandb sweep exp_configs/slottar_craftp.yaml
wandb agent SWEEP_ID
```

For the CompILE results in Table 1. on `Craft (fully)` initialize and run the wandb sweep:
```commandline
wandb sweep exp_configs/compile_craftf.yaml
wandb agent SWEEP_ID
```

For the OMPN results in Table 1. on `Craft (fully)` initialize and run the wandb sweep:
```commandline
wandb sweep exp_configs/ompn_craftf.yaml
wandb agent SWEEP_ID
```

For the analogous results in Tables 2 & 3 on other MiniGrid environments,
please use the corresponding experiment config files (for each model/dataset pair) in the folder `exp_configs`
and run the `wandb sweep` and `wandb agent` commands as in the examples above.

You could also run the training script without the `wandb` dependency by:
```python 
python train.py --dataset_id="craft" --dataset_fname="makeall" --obs_type="full"

python train_baselines.py --model_type="compile" --batch_size=128 --beta=0.1 --hidden_size=128 --latent_size=128 
--dataset_id="craft" --dataset_fname="makeall" --obs_type="full"
```

It will simply print loss and the evaluation metrics to console.

The training script periodically logs variables from our model such as alpha-masks, 
self/slot attention weights, halting probabilities etc. as `npz` files under the appropriate logging directory.
You can re-create the various visualizations shown in the paper by using the helper functions
in `viz.ipynb`. You will need to specify the path to the saved logs (from a training run) 
to create these plots.

## Acknowledgements:
This repository has adapted and/or utilized the following resources:
* The CompILE baseline model in this repository has been re-implemented in tensorflow following the 
 example implementation - https://github.com/tkipf/compile
* The OMPN baseline model in this repository has been re-implemented in tensorflow following the 
 implementation and the offline datasets in the Craft environment using - https://github.com/Ordered-Memory-RL/ompn_craft

### Cite
If you make use of this code in your own work, please cite our paper:
```
@article{gopalakrishnan2022slottar,
  title={Unsupervised Learning of Temporal Abstractions with Slot-based Transformers},
  author={Anand Gopalakrishnan and Kazuki Irie and J{\"u}rgen Schmidhuber and Sjoerd van Steenkiste},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.13573}
}
```
