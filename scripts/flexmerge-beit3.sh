#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Evaluation tasks whose checkpoints should be merged (in the paper we consider only nlvr2,imagenet,coco_captioning,coco_retrieval)
eval_tasks="nlvr2,imagenet,coco_captioning,coco_retrieval"

WANDB_ENTITY="yourentity"
WANDB_PROJECT="yourproject"
# Path to python binary
env_python="path to python"

# Creating the directory for the results
mm_path="path to the root directory of the model-merging repo"
save_dir=mtl
log_dir=$mm_path/results/$save_dir
mkdir -p $log_dir 


# check if open_clip is installed
if ! "$env_python" -c "import open_clip"; then
    echo "==> Installing open_clip"
    "$env_python" -m pip install open-clip-torch
fi

if ! "$env_python" -c "import promptsource"; then
    echo "==> Installing promptsource"
    "$env_python" -m pip install git+https://github.com/bigscience-workshop/promptsource.git
fi

if ! "$env_python" -c "import wandb"; then
    echo "==> Installing wandb"
    "$env_python" -m pip install wandb
fi

beit3_checkpoint_dir="$SCRIPT_DIR/../../../beit3/finetuned_checkpoints"
beit3_eval_config_file="$SCRIPT_DIR/beit3_eval_config.json"

# Path to the data needed for evaluation of BEiT3 models, please follow instructions in the README of beit3/ to download the data
datapath="/path/to/beit3data/ (modify with your own BEiT3 data path)"

model=beit3
run_type=merge
alg=greedy
merge_iter=10000
merge_method=emr
linkage=single

seeds=(90)

for seed in ${seeds[@]}; do
    name=MTL-${run_type}-${alg}-${merge_method}-seed${seed}-${model}
    block_gran=MTL-${model}

    echo "==> Running $name"
    echo "Model dir: $model_dir"
    echo "Block granularity: $block_gran"

    # count time for experiment in hh mm ss
    start=`date +%s`

    "$env_python" "$mm_path/main_merge_beit3.py" \
        --SEED $seed \
        --METHOD MTL_BEIT3 \
        --MODEL $model \
        --MERGE_ITER $merge_iter \
        --RUN_TYPE $run_type \
        --SAVEDIR "$log_dir" \
        --SAVENAME $name \
        --ROUNDS 1 \
        --BLOCK_GRANULARITY $block_gran \
        --MERGE_METHOD $merge_method \
        --ALG $alg \
        --BEIT3_EVAL_TASKS "$eval_tasks" \
        --BEIT3_CHECKPOINT_DIR "$beit3_checkpoint_dir" \
        --BEIT3_EVAL_CONFIG_FILE "$beit3_eval_config_file" \
        --DATAPATH "$datapath" \
        --WANDB_ENTITY $WANDB_ENTITY \
        --WANDB_PROJECT $WANDB_PROJECT \
        --WANDB \
        --GPU_MERGE \
        --USE_CONSTITUENTS_DISTANCE \
        --LINKAGE $linkage
        
    end=`date +%s`
    runtime=$((end-start))
    # Print time in hh:mm:ss
    echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"
done