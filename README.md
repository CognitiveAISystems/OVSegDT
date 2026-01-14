## :hammer_and_wrench: Installation

### Getting Started

Create the conda environment and install all of the dependencies. Mamba is recommended for faster installation:
```bash
conda create -n ovon python=3.8 cmake=3.14.0 -y
conda activate ovon

conda install -n ovon habitat-sim=0.2.3 headless -c conda-forge -c aihabitat -y

conda install -n ovon pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia -c conda-forge -y
conda install -c conda-forge gxx=9
conda install nvidia/label/cuda-11.8.0::cuda-nvcc
conda install -c nvidia cuda-toolkit==11.8
export CUDA_HOME=<PATH_TO_YOUR_CONDA_ENV>

pip install -e .

# Install distributed_dagger and frontier_exploration (From original DagRL codebase: https://github.com/naokiyokoyama/frontier_exploration/tree/a8890d68cfa0d10254238abe9266a76856cb1f17)
cd frontier_exploration && pip install -e . && cd ..

# Install habitat-lab
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines

pip install ftfy regex tqdm GPUtil trimesh seaborn timm scikit-learn einops transformers
pip install git+https://github.com/openai/CLIP.git

# Install semantic input packages:

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install ultralytics


# Patch modeling_llama.py of transformers:

cp modeling_llama.py <<PATH_TO_YOUR_CONDA_ENV>/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py

```
## :dart: Downloading the datasets
First, set the following variables during installation (don't need to put in .bashrc):
```bash
MATTERPORT_TOKEN_ID=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
MATTERPORT_TOKEN_SECRET=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
DATA_DIR=</path/to/ovon/data>
```

### Clone and install habitat-lab, then download datasets
```bash
# Download HM3D 3D scans (scenes_dataset)
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_train_v0.2 \
  --data-path $DATA_DIR &&
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_val_v0.2 \
  --data-path $DATA_DIR
```

The OVON navigation episodes can be found here: https://huggingface.co/datasets/nyokoyama/hm3d_ovon/
The tar.gz file should be decompressed in `data/datasets/ovon/`, such that the `hm3d` directory is located at `data/datasets/ovon/hm3d/`. Delete all "._*" files that appear after decompressing. "val_unseen_easy.json.gz" should be renamed to "val_seen_synonyms.json.gz"

## :weight_lifting: Downloading pre-trained weights
We provide pre-trained checkpoint on the Anonymous Google Drive:

- `ckpt.5.pth`: https://drive.google.com/file/d/1Y26g4xNVFW5UMKmg0IyS1JS-BgBLcoyC/view?usp=sharing

## :arrow_forward: Evaluation within Habitat

Run the following to evaluate:
```bash
python -m ovon.run \
  --run-type eval \
  --exp-config config/experiments/transformer_rl_segm_loss-validation.yaml \
  habitat_baselines.eval_ckpt_path_dir=<path_to_ckpt>
```

By default the evaluation will be performed using YOLOE model on val unseen split of HM3D-OVON. To run experiments on val_seen or val_seen_synonyms, change:

eval.split to "val_seen" or "val_seen_synonyms"
segmentation_source to "yolo_val_seen" or yolo_val_seen_synonyms"

## :rocket: Training

1) Run the following to train with our proposed EALM loss and segmentation loss:

```bash
python -m ovon.run --run-type train \
--debug-datapath \
--exp-config config/experiments/transformer_dagger_ppo_segm_loss.yaml
```

1) Run the following to train with our proposed EALM loss and without segmentation loss:

```bash
python -m ovon.run --run-type train \
--debug-datapath \
--exp-config config/experiments/transformer_dagger_ppo_no_segm_loss.yaml
```


python -m ovon.run --run-type train \
--debug-datapath \
--exp-config config/experiments/transformer_dagger_ppo_segm_loss_demo_zero.yaml


cp /root_home/OVSegDT/modeling_llama.py /opt/conda/envs/habitat/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py


source /opt/conda/etc/profile.d/conda.sh && conda activate habitat
pip install numpy==1.24.4
pip install gym==0.24.1
pip install timm==1.0.15
pip uninstall -y numpy pandas && pip install "numpy<2.0" "pandas<2.0"
