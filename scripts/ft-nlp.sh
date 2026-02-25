#!/bin/bash

PYTHON_PATH=${1:-python3}


if ! $PYTHON_PATH -c "import promptsource"; then
    echo "==> Installing promptsource"
    $PYTHON_PATH -m pip install git+https://github.com/bigscience-workshop/promptsource.git
fi

if ! $PYTHON_PATH -c "import wandb"; then
    echo "==> Installing wandb"
    $PYTHON_PATH -m pip install wandb
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONFIG_DIR="$SCRIPT_DIR/../configs"

datasets=(
    "paws"
    "qasc"
    "quartz"
    "story_cloze"
    "wiki_qa"
    "winogrande"
    "wsc"
)

declare -A val_set_name

val_set_name=(
    ["paws"]="validation"
    ["qasc"]="validation"
    ["quartz"]="validation"
    ["story_cloze"]="validation"
    ["wiki_qa"]="validation"
    ["winogrande"]="validation"
    ["wsc"]="validation"
)

declare -A val_size

val_size=(
    ["paws"]=8000
    ["qasc"]=926
    ["quartz"]=384
    ["story_cloze"]=1871
    ["wiki_qa"]=2733
    ["winogrande"]=1267
    ["wsc"]=104
)

checkpoint_frequencies=(
    ["paws"]=100
    ["qasc"]=100
    ["quartz"]=100
    ["story_cloze"]=100
    ["wiki_qa"]=100
    ["winogrande"]=100
    ["wsc"]=5
)

# Depending on the container where this script is done modify this accordingly
export DATASETS_CACHE_DIR="/mnt/nfs/shared"

# Uncomment the line corresponding to the desired base model
# config_file="$CONFIG_DIR/t5_base-a100-generic.json"
config_file="$CONFIG_DIR/t5_large-a100-generic.json"
training_script_path="$SCRIPT_DIR/../NLP/training.py"

echo "Using python path: $PYTHON_PATH"

for dataset_name in "${datasets[@]}"
do
    val_subset_name=${val_set_name[$dataset_name]}
    val_subset_size=${val_size[$dataset_name]}
    checkpoint_frequency=${checkpoint_frequencies[$dataset_name]}

    echo "Training solo model for $dataset_name"
    echo "Validation subset: $val_subset_name (size: $val_subset_size)"

    
    $PYTHON_PATH $training_script_path \
        -c $config_file \
        -k project_name=training experiment_name=$dataset_name train_dataset=$dataset_name inference_dataset=$dataset_name split=$val_subset_name num_val_samples=$val_subset_size checkpoint_frequency=$checkpoint_frequency
done