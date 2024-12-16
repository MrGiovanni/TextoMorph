## ⚙️ Requirements
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
