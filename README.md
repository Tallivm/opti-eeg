*Expect this repository to get breaking updates throughout the spring, as it is used for certain academic projects.*

# Introduction

`opti-eeg` is a set of pipelines used for tuning deep learning models for electroencephalography (EEG) data. It allows the user to set ranges of parameters for tuning via [Optuna](https://optuna.readthedocs.io/en/stable/), and uses [WanDB](https://wandb.ai/) for logging. Main features:
- Models with best validation accuracy are saved into .ckpt files, so you can reuse them later. Model filenames are generated from timestamps to prevent any overwriting.
- There are two training early stop checks: if the model does not improve for a set number of epochs, and if the model overfits (with a set threshold of the difference between train and alidation accuracy).
- In-built cross-validation (CV): all CV metrics are recorded, so you can extract the best model from all folds. In WanDB, CV runs are grouped to see the average metrics. The test split is separate from CV and is used only at the end of training process with the best model.
- Model names can be added as a set of tuned parameter values, which allows to train several model architectures and compare them in the same project.
- Other parameters which can be tuned are optimizer, scheduler, activation, etc. It should also be possible to tune with several different datasets, e.g. with different preprocessing, to see which results in higher evaluation.
- Evaluation metric is easily customizable.
- There is an additional script for data preprocessing, with options to filter and cut data into epochs, as well as join several data files into a single dataset (e.g. if EEG and EOG were recorded separately, with automatic resampling if needed).
- Other than WanDB records, all outputs are logged and stored in the `logs` directory.
- `--noopt` flag will disable tuning and instead complete a single training run with parameters indicated in a sepatate config.
- All currently included models are [EEGNet](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c) variations.

# Installation

Prerequisites:
- Python 3.12 or newer
- For GPU acceleration: graphics card with CUDA (NVIDIA) or ROCm support (AMD). Works without it as well but may be slower.

1. Create and activate a new environment using your favorite environment manager, e.g. `venv` or `conda`.
1. Install packages from "requirements.txt". E.g. with pip: `pip install -r requirements.txt`.
1. Install [PyTorch](https://pytorch.org/get-started/locally/) into the same environment.
   - Carefully choose the version which fits your system (OS and CUDA/ROCm version)!
   - `torchvision` and `torchaudio` are not needed, `torch` is enough.
   - Make sure you have enough space - a standalone venv environment with CUDA may take ~8GB of space.


# Usage

1. Use "config_template.json" as a template to build your config. See the section below for more information.
1. Check "optuna_params.py" file for the sweep parameter setup.
   - It currently uses four different EEGNet versions (classic `EEGNet`; its variation `EEGNetConv` with a CNN classification layer; `TSGLEEGNet` adapted from [this repo](https://github.com/JohnBoxAnn/TSGL-EEGNet); and `EEGNetProto` which is my variation with a KNN layer). All models are based on the `EEGNet_Modular` class ("models/eegnet_modular.py").
1. Change directory to `src/opti-eeg`, e.g. by running in terminal: `cd src/opti-eeg`.
1. Run `python run_optieeg.py config_template.json` to start the optimization. Run `python run_optieeg.py config_template.json --nowandb` to disable WanDB if e.g. you don't have an account.


# Creating a config

The config contains all information relevant for hyperparameter optimization.

**The template won't work as-is** - these parameters should be filled in:
- "project_name" - a new WanDB project with this name will be created.
- "wandb_user_key" (inside "opt") - only needed if WanDB is not disabled.
- "data_path" and "labels_path" (inside "data") - paths to data files; currently only .npy files are supported.
- Other parameters in the template are fit for the [DEED](http://www.deeddataset.com/) dataset. Make sure to change them so that they would correspond to your dataset.


**Full list of options:**

- `"project_name"`: project name which will be used by Optuna and WanDB
- `"path"`:
   - `"checkpoint_path"`: path to save model checkpoints (.ckpt files).
   - `"data_path"`: path to data which is used for predictions.
   - `"labels_path"`: path to data which is predicted. Labels can be numbers or strings.
   - `"test_data_path"`: if provided, path to data which will be used only for testing; otherwise, test will be randomly taken from "data_path".
   - `"test_labels_path"`: if provided, path to data labels which will be used only for testing the model; otherwise, test will be randomly taken from "labels_path".
   - `"fixed_config_path"`: only used if `--noopt` flag is provided. Useful when you need to obtain the bbest model for a certain set of parameters.
- `"opt"`:
   -  `"wandb_user_key"`: your WanDB key.
   -  `"wandb_number_of_failures"`: after how many WanDB failures the process should stop.
   -  `"optuna_n_trials"`: how many runs in total to do.
   -  `"create_new_db"`: if `true`, a new Optuna Database will be created even if old exists already.
- `"data"`:
   - `"sample_rate"`: EEG sample rate in Hz.
   - `"n_classes"`: number of unique labels to predict.
   - `"class_names"`: ordered names of labels, should be same length as `"n_classes"`. If you want to drop a class, name it `""`. Any duplicated names will trigger automatic label remapping (e.g. `["table", "chair", "chair"]` will result in 2 classes after remapping).
   - `"n_channels"`: number of EEG channels.
   - `"channel_names"`: ordered names of EEG channels, should be same length as `"n_channels"`.
   - `"omit_channels"`: EEG channels to omit from predictions (can be left empty), useful if you want to ignore mastoids and such.
- `"train"`:
  -  `"max_n_epochs"`: the max number of epochs per run
  -  `"n_cv"`: number of cross-validation splits
  -  `"loss_func_name"`: name of a loss function to use
  -  `"eval_metric_name"`: name of a main metric which is optimized
  -  `"eval_increases"`: `true` - bigger metric number means better result
  -  `"patience"`: early stop - after how many epochs without improvement the run should stop
  -  `"min_delta"`: early stop - minimum required evaluation metric improvement to reset the early stopping counter
  -  `"overfit_thresh"`: early stop - minimum required difference between train and validation metric to stop
  -  `"overfit_epoch"`: early stop - from which epoch overfit is measured
  -  `"data_seed"`: seed used for data splitting
  -  `"model_seed"`: seed used for model initialization


# Preparing data for the pipeline

The "run_data_preprocessing.py" script is used to prepare data for the pipeline. It uses [Breesy](https://readream.net/Breesy/intro.html) to read all files from provided folder, using a certain pattern in case some information about recordings is stored in the filenames.
- For example, files from the DEED dataset use this pattern: `"G_{s}_M{m}_E{e}_R{r}_{prev_phase}_raw_ref.mat"`. Here, `{s}` - subject ID, `{m}` - movie ID, `{e}` - emotion label, `{r}` - order of REM, and `{prev_phase}` - sleep phase before this REM one.
- To create your own pattern, use the `{e}` placeholder, and if there are more variable places in filenames, you should put placeholders in those places with anything different inside. Currently, the script only pays attention to the `{e}` amd `{s}` placeholders.


# Code cuztomization

- You can add new optimizers (`scripts/optimizers.py`), schedulers (`scripts/schedulers.py`), activations (`scripts/activations.py`), poolings (`scripts/poolings.py`), regularizers (`scripts/regularizers.py`), loss functions (`scripts/losses.py`), evaluation metrics (`scripts/eval_metrics.py`), as lond as they have the same usage pattern as the ones already there.
- You can add new EEGNet variations into `models/eegnet_modular.py`.
