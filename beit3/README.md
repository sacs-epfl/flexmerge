# Multi-Modal Models

This folder contains the code for fine-tuning and evaluating the BEiT-3 model, adapted from [the Microsoft UniLM repository](https://github.com/microsoft/unilm/tree/master/beit3).


## Setup

For the rest of section it is assumed you are located at the root of `beit3` directory.

1. **Install Python dependencies**  

The dependencies required to run this code are not fully included in `environment.yml` at the root of the repository, so also install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

2. **Fix protobuf version (if needed)**

If you encounter compatibility errors with `protobuf` run the following command to downgrade the package:
```bash
pip install 'protobuf<=3.20.1' --force-reinstall
```

3. **Install Java Runtime and other required packages**

Certain evaluation scripts require a JRE. On Debian/Ubuntu you can install it using:
```bash
apt update && apt install -y default-jre
```

The code also requires some more packages to run, so they have to be installed too:
```bash
apt update && apt install -y libgl1 libglib2.0-0
```

## Downloading required data

Once environment is properly setup, you have to download checkpoints used for experiments, as well as datasets used for their evaluation. Most of this process is streamlined by our scripts if your system supports `bash` shell. 

### Downloading Checkpoints & Tokenizer
First, run:
```bash
./finetuned_checkpoints/download_checkpoints.sh
```
This will download the pretrained BEiT-3 base checkpoints and tokenizer (~5 minutes).

### Preparing Evaluation Data

Downloading of datasets can be done either manually or using scripts we provide. We advise using our scripts if they are supported on your system as manually downloading datasets is time consuming.

#### Manual downloading of datasets

Create a single root data directory; in this example we’ll use `/path/to/beit3data/`. Under it, set up:
```
/path/to/beit3data/
├── coco/
├── imagenet/
└── nlvr2/
```

- **COCO**: place COCO Captioning, COCO Retrieval and VQAv2 test splits under `coco/`.
- **ImageNet-1k**: place the test split under `imagenet/`.
- **NLVR2**: place the test split under `nlvr2/`. You must request access via [this Google form](https://goo.gl/forms/yS29stWnFWzrDBFH3).

For all dataset-specific instructions (folder structure, file names) follow the UniLM guide:
https://github.com/microsoft/unilm/tree/master/beit3/get_started.

#### Automated Download Script

You can also fetch all required data with our helper script. From `beit3/` directory run:
```bash
./scripts/download_beit3_eval_data.sh /path/to/beit3data/
```

**Note**:
- This will download tens of gigabytes, hence this script may take a while.
- External hosting may change causing the script not to be able to find resource; if the script fails, download manually per [the UniLM instructions](https://github.com/microsoft/unilm/tree/master/beit3/get_started).


## Running experiments

Once everything is set up, you can run FlexMerge with EMR Merging on the BEiT-3 model via the provided script. No additional parameters are required: 
```bash
/scripts/flexmerge-beit3.sh
```