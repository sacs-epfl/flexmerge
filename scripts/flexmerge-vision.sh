#!/bin/bash

##################### UPDATE THIS PATH #####################
root_dir=<path_to_root_dir_of_the_codebase>
log_dir=$root_dir/results/
mkdir -p $log_dir 

##################### UPDATE THIS PATH #####################
env_python=<path_to_python_env>

##################### UPDATE THIS PATH #####################
model_dir=<path_to_where_finetuned_checkpoints_are_saved>
data_dir=<path_to_where_datasets_are_saved>

# check if open_clip is installed
if ! $env_python -c "import open_clip"; then
    echo "==> Installing open_clip"
    $env_python -m pip install open-clip-torch
fi

size=8
run_type=merge
model=ViT-B-32
alg=greedy
merge_iter=1000
block_gran=MTL-transformer+PP
merge_method=ta

seeds=(90)

for seed in ${seeds[@]}; do
    
    name=MTL-${run_type}-${alg}-${merge_method}-n${size}-seed${seed}-${model}

    # count time for experiment in hh mm ss
    start=`date +%s`

    $env_python $root_dir/main_merge.py \
        --DATAPATH $data_dir \
        --SEED $seed \
        --METHOD MTL \
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
        --WANDB \
        --LAMBDA 1.5 \
        --USE_CONSTITUENTS_DISTANCE \
        --LINKAGE single
        
    end=`date +%s`
    runtime=$((end-start))
    # Print time in hh:mm:ss
    echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"

done