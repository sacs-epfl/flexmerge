#!/bin/bash

WANDB_ENTITY="yourentity"
WANDB_PROJECT="yourproject"
# Path to the parent directory of the model directories. For instance in this directory there should be t5-base and t5-large directories.
#   And in each of these directories there should be a model trained on each of the 7 tasks.
#       MODEL_BASE_DIR
#       ├── t5-base
#       │   ├── paws
#       │   ├── qasc
#       │   ├── quartz
#       │   ├── story_cloze
#       │   ├── wiki_qa
#       │   ├── winogrande
#       │   └── wsc
#       └── t5-large
#           ├── paws
#           ├── qasc
#           ├── quartz
#           ├── story_cloze
#           ├── wiki_qa
#           ├── winogrande
#           └── wsc
MODEL_BASE_DIR="path to model dirs"
# Path to python binary
env_python="path to python"

mm_path=$HOME_DIR/model-merging
save_dir=mtl
log_dir=$mm_path/results/$save_dir
mkdir -p $log_dir 

# check if open_clip is installed
if ! $env_python -c "import open_clip"; then
    echo "==> Installing open_clip"
    $env_python -m pip install open-clip-torch
fi

if ! $env_python -c "import promptsource"; then
    echo "==> Installing promptsource"
    $env_python -m pip install git+https://github.com/bigscience-workshop/promptsource.git
fi

if ! $env_python -c "import wandb"; then
    echo "==> Installing wandb"
    $env_python -m pip install wandb
fi

size=7
run_type=merge
alg=greedy
merge_iter=10000
merge_method=avg

seeds=(90)
models=("t5-base" "t5-large")
declare -A config_filename
config_filename=(
    ["t5-base"]="t5_base-a100-generic.json"
    ["t5-large"]="t5_large-a100-generic.json"
)

export DATASETS_CACHE_DIR="/mnt/nfs/shared"

for seed in ${seeds[@]}; do
    for model in ${models[@]}; do
        name=MTL-${run_type}-${alg}-${merge_method}-n${size}-seed${seed}-${model}
        model_dir=$MODEL_BASE_DIR/$model
        block_gran=MTL-${model}

        echo "==> Running $name"
        echo "Model dir: $model_dir"
        echo "Block granularity: $block_gran"

        # count time for experiment in hh mm ss
        start=`date +%s`

        $env_python $mm_path/main_merge_nlp_fft.py \
            --DATAPATH $HOME \
            --SEED $seed \
            --METHOD MTL_FFTNLP \
            --BATCH_SIZE 64 \
            --MODEL $model \
            --MERGE_ITER $merge_iter \
            --RUN_TYPE $run_type \
            --SAVEDIR $log_dir \
            --SAVENAME $name \
            --MODELDIR $model_dir \
            --NUM_CLIENTS $size \
            --ROUNDS 1 \
            --GPU_MERGE \
            --BLOCK_GRANULARITY $block_gran \
            --MERGE_METHOD $merge_method \
            --ALG $alg \
            --FFT_CONFIGFILE $mm_path/configs/${config_filename[$model]} \
            --WANDB \
            --WANDB_ENTITY $WANDB_ENTITY \
            --WANDB_PROJECT $WANDB_PROJECT \
            --USE_CONSTITUENTS_DISTANCE \
            --LINKAGE single
        
        end=`date +%s`
        runtime=$((end-start))
        # Print time in hh:mm:ss
        echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"
    done
done