#!/bin/bash

##################### UPDATE THIS PATH #####################
root_dir=<path_to_root_dir_of_the_codebase>

##################### UPDATE THIS PATH #####################
env_python=<path_to_python_env>

##################### UPDATE THIS PATH #####################
model_dir=<path_to_where_finetuned_checkpoints_are_saved>
data_dir=<path_to_where_datasets_are_saved>

# choose from any of the 20 datasets
dataset=EuroSAT
# choose from ViT-B-32 or ViT-L-14
model=ViT-B-32
seeds=(90)

# check if open_clip is installed
if ! $env_python -c "import open_clip"; then
    echo "==> Installing open_clip"
    $env_python -m pip install open-clip-torch
fi


for seed in ${seeds[@]}; do
    
    name=FT-$dataset

    # count time for experiment in hh mm ss
    start=`date +%s`

    $env_python $root_dir/finetune.py \
        --DATATYPE $dataset \
        --DATAPATH $data_dir \
        --MODELDIR $model_dir \
        --MODEL $model \
        --SAVENAME $name

    end=`date +%s`
    runtime=$((end-start))
    
    # Print time in hh:mm:ss
    echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"

done