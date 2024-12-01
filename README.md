# Text-Driven Tumor Synthesis
Tumor synthesis can generate examples that AI often misses or over-detects, improving AI performance by training on these challenging cases. However, existing synthesis methods, which are typically unconditional---generating images from random variables---or conditioned only by tumor shapes, lack controllability over specific tumor characteristics such as texture, heterogeneity, boundaries, and pathology type. As a result, the generated tumors may be overly similar or duplicates of existing training data, failing to effectively address AI's weaknesses. We propose a new text-driven tumor synthesis approach, termed **TextoMorph**, that provides textual control over tumor characteristics.
<div align="center">
  <img src="https://github.com/user-attachments/assets/b10d59ee-78ca-443f-9d5a-862d0235a5c7" alt="fig_cloudplot" width="600"/>
</div>

[Xinran Li](https://scholar.google.com/citations?hl=zh-CN&user=awRZX_gAAAAJ)<sup>1,2</sup>, [Yi Shuai](https://openreview.net/profile?id=~Yi_Shuai1)<sup>3</sup>, [Chen Liu](https://scholar.google.com/citations?user=i938yiEAAAAJ&hl=zh-CN)<sup>1,4</sup>, [Qi Chen](https://scholar.google.com/citations?user=4Q5gs2MAAAAJ&hl=en)<sup>1,5</sup>, [Qilong Wu](https://github.com/JerryWu-code)<sup>1,6</sup>, Pengfei Gao<sup>7</sup>, Dong Yang<sup>7</sup>,  
Can Zhao<sup>7</sup>, [Pedro R. A. S. Bassi](https://scholar.google.com/citations?hl=zh-CN&user=NftgL6gAAAAJ)<sup>1,8,9</sup>, [Daguang Xu](https://research.nvidia.com/person/daguang-xu)<sup>7</sup>, [Kang Wang](https://radiology.ucsf.edu/people/kang-wang)<sup>10</sup>, [Yang Yang](https://scholar.google.com/citations?user=6XsJUBIAAAAJ&hl=zh-CN)<sup>10</sup>, [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>1</sup>, [Zongwei Zhou](https://www.zongweiz.com/)<sup>1,*</sup>  

<sup>1</sup> Johns Hopkins University  
<sup>2</sup> Shenzhen Technology University  
<sup>3</sup> Sun Yat-sen University  
<sup>4</sup> Hong Kong Polytechnic University  
<sup>5</sup> University of Chinese Academy of Sciences  
<sup>6</sup> National University of Singapore  
<sup>7</sup> NVIDIA  
<sup>8</sup> University of Bologna  
<sup>9</sup> Italian Institute of Technology  
<sup>10</sup> University of California, San Francisco  



**We have documented common questions for the paper in [Frequently Asked Questions (FAQ)](documents/FAQ.md).**

**We have summarized publications related to tumor synthesis in [Awesome Synthetic Tumors](https://github.com/MrGiovanni/SyntheticTumors/blob/main/AWESOME.md) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re).**


## ⚙️ Requirements
Begin by cloning the project repository to your local machine using the following command:

```bash
git clone https://github.com/MrGiovanni/TextoMorph.git
cd TextoMorph
```
Create a virtual environment using the following command:

```bash
conda create -n TextoMorph python=3.8
source activate TextoMorph # or conda activate TextoMorph
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
---
## 📂 Dataset Download Instructions

This document provides step-by-step instructions to download and prepare datasets required for the project.
### 📥 Download Unhealthy Data
- 📌 Liver Tumor Segmentation Challenge (LiTS)
  🌐 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094)  

- 📌 MSD-Pancreas
  🌐 [MSD-Pancreas](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)

- 📌 KiTS
  🌐 [KiTS](https://kits-challenge.org/kits23/#download-block)

Run the following commands to download and extract unhealthy datasets:
```bash
wget https://huggingface.co/datasets/qicq1c/Pubilcdataset/resolve/main/10_Decathlon/Task03_Liver.tar.gz # Task03_Liver.tar.gz (28.7 GB)
wget https://huggingface.co/datasets/qicq1c/Pubilcdataset/resolve/main/10_Decathlon/Task07_Pancreas.tar.gz # Task07_Pancreas.tar.gz (28.7 GB)
wget https://huggingface.co/datasets/qicq1c/Pubilcdataset/resolve/main/05_KiTS.tar.gz # KiTS.tar.gz (28 GB)
```
Extract the downloaded files:
```bash
tar -zxvf Task03_Liver.tar.gz
tar -zxvf Task07_Pancreas.tar.gz
tar -zxvf 05_KiTS.tar.gz
```
### 📥 Download Healthy Data
- 📌 AbdonmenAtlas 1.1
  🌐 [AbdonmenAtlas 1.1](https://github.com/MrGiovanni/AbdomenAtlas)
- 📌 Healthy CT Dataset
  🌐 [HealthyCT Dataset](https://huggingface.co/datasets/qicq1c/HealthyCT)  

Download **AbdonmenAtlas 1.0** using the following commands:
```bash
huggingface-cli BodyMaps/_AbdomenAtlas1.1Mini --token paste_your_token_here --repo-type dataset --local-dir .
bash unzip.sh
bash delete.sh
```
Download and prepare the **HealthyCT** dataset:
```bash
huggingface-cli download qicq1c/HealthyCT  --repo-type dataset --local-dir .  --cache-dir ./cache
cat healthy_ct.zip* > HealthyCT.zip
rm -rf healthy_ct.zip* cache
unzip -o -q HealthyCT.zip -d /HealthyCT
```


## STEP 1. 🚀 Train Diffusion Model
### 📥 Download Required Files
We offer the pre-trained checkpoint of the **Autoencoder Model**, which was trained on the **AbdomenAtlas 1.1 dataset** (see details in [SuPreM](https://github.com/MrGiovanni/SuPreM)).  
This checkpoint can be directly used for the Diffusion model if you do not want to re-train the Autoencoder Model. Simply download it to `Diffusion/pretrained_models/AutoencoderModel.ckpt` by running the following command:

```bash
cd Diffusion/pretrained_models/
wget https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/AutoencoderModel/AutoencoderModel.ckpt
```
### 🔧 Start training.
```bash
cd Diffusion/
vqgan_ckpt=<pretrained-AutoencoderModel> (e.g., /pretrained_models/AutoencoderModel.ckpt)
datapath=<your-datapath> 
tumorlabel=<your-labelpath> 
python train.py dataset.name=liver_tumor_train dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['liver'] dataset.uniform_sample=False model.results_folder_postfix="liver"  model.vqgan_ckpt=$vqgan_ckpt
```

We offer the pre-trained checkpoints of Diffusion Model, which were trained for early-stage and mid-/late- stage tumors for liver, pancreas and kidney, respectively. This checkpoint can be directly used for STEP3 if you do not want to re-train the Diffusion Model. Simply download it to `Segmentation/TumorGeneration/model_weight`

### 🔗 Checkpoints Overview

The following checkpoints are available for download. These pre-trained weights can be used directly for tumor generation and segmentation tasks for different organs:

| Tumor      | Download                                                                                      |
|------------|-----------------------------------------------------------------------------------------------|
| **Liver**  | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/liver.pt) |
| **Pancreas** | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/pancreas.pt) |
| **Kidney**   | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/kidney.pt?download=true) |

## STEP 2. 🚀 Train Segmentation model

To use the **TumorGeneration** segmentation model, download the necessary pre-trained weights.  
Follow the instructions below to ensure the proper setup of your directory structure and model files.

### 📥 Download Required Files

Run the following commands to download the pre-trained weights:

```bash
# Navigate to the model weights directory
cd Segmentation/TumorGeneration/model_weight

# Download the Autoencoder Model checkpoint
wget https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/AutoencoderModel/AutoencoderModel.ckpt

# Download tumor-specific checkpoints
wget https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/liver.pt
wget https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/pancreas.pt
wget https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/kidney.pt

cd ../..
```
### 🔧 Start training.
```bash
cd Segmentation

healthy_datapath=<your-datapath>
datapath=<your-datapath>
cache_rate=1.0
batch_size=12
val_every=50
workers=12
organ=liver
fold=0
backbone=unet
logdir="runs/$organ.fold$fold.$backbone"
datafold_dir=cross_eval/"$organ"_aug_data_fold/
dist=$((RANDOM % 99999 + 10000))
python -W ignore main.py --model_name $backbone --cache_rate $cache_rate --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir
```
### 🔗 Checkpoint Overview

| **Model**           | **Download**                                                                                 |
|----------------------|---------------------------------------------------------------------------------------------------|
| **Liver**     | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/liver.pt) |
| **Pancreas**   | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/pancreas.pt) |
| **Kidney**     | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/kidney.pt) |

## STEP 3. Evaluation


```bash
cd Segmentation
datapath=<your-datapath>
organ=liver
fold=0
datafold_dir=cross_eval/"$organ"_aug_data_fold/
python -W ignore validation.py --model=unet --data_root $datapath --datafold_dir $datafold_dir --tumor_type tumor --organ_type $organ --fold $fold --log_dir $organ/$organ.fold$fold.unet --save_dir out/$organ/$organ.fold$fold.unet
```

## 🛠️ Using the Singularity Container for TextoMorph

We provide a **Singularity container** for running **TextoMorph** tasks, which supports both text-driven tumor synthesis and segmentation (organ, tumor). Follow the instructions below to get started.


### 1️⃣ Text-Driven Tumor Synthesis

To generate tumors based on textual descriptions, use the following command:

```bash
inputs_data=/path/to/your/healthyCT
inputs_label=liver          # Example: pancreas, kidney
text="The liver contains arterial enhancement and washout."
outputs_data=/path/to/your/output/Text-Driven-Tumor

SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity run --nv -B $inputs_data:/workspace/inputs -B $outputs_data:/workspace/outputs textomerph.sif
```
### 1️⃣ Segmentation (Organ, Tumor)
To perform organ or tumor segmentation on CT scans, use the following command:
```bash

inputs_data=/path/to/your/CT/scan/folders
outputs_data=/path/to/your/output/folders

SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity run --nv -B $inputs_data:/workspace/inputs -B $outputs_data:/workspace/outputs textomerph.sif
```
## Acknowledgments
This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the Patrick J. McGovern Foundation Award.
