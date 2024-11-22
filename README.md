## STEP 1. Create a virtual environment 

The environment is the same as [DiffTumor](https://github.com/MrGiovanni/DiffTumor)

```bash
conda create -n difftumor python=3.8
source activate difftumor # or conda activate difftumor
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
## STEP 2. Download Datasets
download unhealthy data
```bash
wget https://huggingface.co/datasets/qicq1c/Pubilcdataset/resolve/main/10_Decathlon/Task03_Liver.tar.gz # Task03_Liver.tar.gz (28.7 GB)
wget https://huggingface.co/datasets/qicq1c/Pubilcdataset/resolve/main/10_Decathlon/Task07_Pancreas.tar.gz # Task07_Pancreas.tar.gz (28.7 GB)
wget https://huggingface.co/datasets/qicq1c/Pubilcdataset/resolve/main/05_KiTS.tar.gz # KiTS.tar.gz (28 GB)
```
```bash
tar -zxvf Task03_Liver.tar.gz
tar -zxvf Task07_Pancreas.tar.gz
tar -zxvf 05_KiTS.tar.gz
```
download healthy data

5001-9262
```bash
wget https://huggingface.co/datasets/BodyMaps/_AbdomenAtlas1.1Mini/resolve/main/AbdomenAtlas1.1Mini_BDMAP_00005001_00005500.tar.gz?download=true
wget https://huggingface.co/datasets/BodyMaps/_AbdomenAtlas1.1Mini/resolve/main/AbdomenAtlas1.1Mini_BDMAP_00005501_00006000.tar.gz?download=true
wget https://huggingface.co/datasets/BodyMaps/_AbdomenAtlas1.1Mini/resolve/main/AbdomenAtlas1.1Mini_BDMAP_00006001_00006500.tar.gz?download=true
wget https://huggingface.co/datasets/BodyMaps/_AbdomenAtlas1.1Mini/resolve/main/AbdomenAtlas1.1Mini_BDMAP_00006501_00007000.tar.gz?download=true
wget https://huggingface.co/datasets/BodyMaps/_AbdomenAtlas1.1Mini/resolve/main/AbdomenAtlas1.1Mini_BDMAP_00007001_00007500.tar.gz?download=true
wget https://huggingface.co/datasets/BodyMaps/_AbdomenAtlas1.1Mini/resolve/main/AbdomenAtlas1.1Mini_BDMAP_00007501_00008000.tar.gz?download=true
wget https://huggingface.co/datasets/BodyMaps/_AbdomenAtlas1.1Mini/resolve/main/AbdomenAtlas1.1Mini_BDMAP_00008001_00008500.tar.gz?download=true
wget https://huggingface.co/datasets/BodyMaps/_AbdomenAtlas1.1Mini/resolve/main/AbdomenAtlas1.1Mini_BDMAP_00008501_00009000.tar.gz?download=true
wget https://huggingface.co/datasets/BodyMaps/_AbdomenAtlas1.1Mini/resolve/main/AbdomenAtlas1.1Mini_BDMAP_00009001_00009262.tar.gz?download=true
```

```bash
huggingface-cli BodyMaps/_AbdomenAtlas1.1Mini --token paste_your_token_here --repo-type dataset --local-dir .
bash unzip.sh
bash delete.sh
```
## STEP 3. Installation
```bash
git clone https://github.com/Alena-Xinran/Descriptivetumor.git
cd Descriptivetumor/
```
## STEP 4. train
```bash
sh hg.sh
```
