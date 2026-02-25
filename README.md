# FlexMerge

<img src="flexmerge.drawio.svg" width="75%">

This respository contains the source code for our paper "Navigating the accuracy-size trade-off with flexible model merging", accepted at ICLR 2026.

Our code is built on top of existing codebases of the following papers:
- [TIES-Merging](https://github.com/prateeky2806/ties-merging)
- [Consensus](https://github.com/nik-dim/tall_masks)
- [EMR-Merging](https://github.com/harveyhuang18/EMR_Merging)
- [PCB-Merging](https://github.com/duguodong7/pcb-merging)

### Create python environment

```bash
conda env create -f environment.yml
```

Our code has only been tested on A100 GPUs. If you encounter any wandb errors, please try rerunning a few more times. 

### Before running the code

Please modify the file `./Utils/args_utils.py` by changing the default values of the parameters `WANDB_ENTITY` and `WANDB_PROJECT` to your desired values.

## Setting up datasets

### (1) NLP (PEFT and full parameter FT) datasets

In most cases, the code will automatically download the required dataset. However, an exception to this is the Story Cloze dataset, which must be manually downloaded and placed in a desired location. Then, set the environment variable `DATASETS_CACHE_DIR` to the parent directory where the dataset is stored.

For example, if the dataset is stored at `/home/user/story_cloze`, you should set:
```bash
export DATASETS_CACHE_DIR="/home/user"
```

For more information please refer to [this repository](https://github.com/prateeky2806/ties-merging).

### (2) Vision datasets

Please follow the instructions provided on the GitHub of [Consensus](https://github.com/nik-dim/tall_masks) to set up the vision datasets.

### (3) Multi-modal datasets

All the instructions for the setup can be found in the ReadMe located `beit3/README.md`.

## Fine-tuning models

### (1) Vision models

Vision models can be fine-tuned using the script available at `./scripts/vision-ft.sh`.

```bash
./scripts/ft-vision.sh
```

You must set the following paths in the script before running it:

```bash
##################### UPDATE THIS PATH #####################
root_dir=<path_to_root_dir_of_the_codebase>

##################### UPDATE THIS PATH #####################
env_python=<path_to_python_env>

##################### UPDATE THIS PATH #####################
model_dir=<path_to_where_finetuned_checkpoints_are_saved>
data_dir=<path_to_where_datasets_are_saved>
```

### (2) PEFT models

We use the checkpoints provided by the authors of the [TIES-Merging](https://proceedings.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf) paper directly. Please check their GitHub repository for more information.

### (3) NLP models

NLP models can be fine-tuned using the script available at `./scripts/nlp-ft.sh` by running:

```bash
bash ./scripts_to_submit/ft-nlp.sh [optional: path to python binary]
```

This script assumes all required packages are installed and will fine-tune the given base model for all datasets.
To change the base model, modify the `config_file` variable in the bash script as follows:

```bash
# Uncomment the line corresponding to the desired base model
# config_file="$CONFIG_DIR/t5_base-a100-generic.json"
config_file="$CONFIG_DIR/t5_large-a100-generic.json"
```

### (4) Multi-modal models

All the instructions for the setup can be found in the ReadMe located `beit3/README.md`.

## Fine-tuned models' accuracy files

Before running FlexMerge, you need to create the accuracy files for the fine-tuned models (manually) and store them in the designated position as described below. These accuracy files are necessary for evaluation of the normalized accuracy, which is defined w.r.t. fine-tuning accuracy of each task.

As an example, we provide the accuracy files with the exact names as required in the folder `sample_accuracy_files`. These files correspond to accuracy obtained by the models we fine-tune and use in our experiments. The only relevant row in these csv files is the one corresponding to `Method == Individual`.

### Placement of accuracy files

The code expects that the files with exact names as in the `sample_accuracy_files` folder are placed in the same location as the `${model_dir}` where all fine-tuned models are stored. The following exact directory structure is expected:

```
# Vision models
${model_dir}
    ├── ViT-B-32
        ├── MNISTVal
            |––– nonlinear_finetuned.pt
            |––– nonlinear_zeroshot.pt
        ├── CIFAR10Val
        ├── ...
    ├── ViT-L-14
    ├── ft-accuracy-n8-ViT-B-32.csv
    ├── ft-accuracy-n8-ViT-L-14.csv
    |── ft-accuracy-n30-ViT-B-32.csv
    |── ft-accuracy-n30-ViT-L-14.csv

# NLP models
${model_dir}
    ├── t5-base
        ├── ft-accuracy-t5-base.csv
        ├── quartz
            |––– best_model.pt
            |––– ...
        ├── story_cloze
            |––– best_model.pt
            |––– ...
        ├── ...
    ├── t5-large
        ├── ft-accuracy-t5-large.csv
        ├── quartz
            |––– best_model.pt
            |––– ...
        ├── ...

# PEFT models
${model_dir}
    ├── ft-accuracy-n11-T0_3B.csv
    ├── ia3
        ├── story_cloze
            |––– best.pt
            |––– ...
        ├── anli-r1
            |––– best.pt
            |––– ...
        ├── ...
```

## Running FlexMerge

### (1) Vision models
Merging of the fine-tuned vision models can be performed by running the following script:

```bash
./scripts/flexmerge-vision.sh
```

The default script uses Task Arithmetic (`ta`) as the merging method. If you prefer a different method, modify the `merge_method` variable to either `avg`, `ties`,`consensus` or `emr`.
You can set the model to either `ViT-B-32` or `ViT-L-14` by modifying the `model` variable and change the total number of tasks by modifying the `size` variable from {4, 8, 20}.

### (2) PEFT models
Merge the PEFT (IA)^3 models by running the following script:

```bash
./scripts/flexmerge-peft.sh
```

The default script uses TIES as the merging method. If you prefer a different method, modify the `merge_method` to your preferred choice.

### (3) NLP models

Merging of the fine-tuned NLP models can be performed by running the following script:

```bash
bash ./scripts/flexmerge-nlp.sh
```

Before running the script, ensure that you have set the following variables in the script file to the correct values for your experiments:  `WANDB_ENTITY`, `WANDB_PROJECT`, `MODEL_BASE_DIR` and `env_python`. This script uses averaging as the default merging method. If you prefer a different method, modify the `merge_method` variable:
- Set it to `ta` for Task Arithmetic.
- Set it to `ties` for the TIES algorithm.

All the scripts mentioned above assume that all required Python packages are already installed.

### (4) Multi-modal models
Merging of the fine-tuned multi-modal models can be performed by running the following script:

```bash
./scripts/flexmerge-beit3.sh
```
The default script uses EMR-Merging as the merging method.

## Coming soon

The currently released code computes cosine similarity newly in each iteration, making it slightly slow. However, in wall clock time, the overhead is small and dominated mostly by evaluation of the generated model on the test sets at different size intervals. We are currently working on releasing the optimized version where cosine similarity is pre-computed only once.
