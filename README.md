# Text-Driven Tumor Synthesis
<div align="center">
  <img src="/utils/fig_cloudplot.png" alt="fig_cloudplot"/>
</div>
Current tumor synthesis methods lack precise control over specific characteristics, resulting in similar or duplicate examples that fail to effectively enhance AI performance. We introduce TextoMorph, a novel text-driven approach that allows detailed textual control over tumor features such as texture, heterogeneity, boundaries, and pathology type to improve AI training.

## Paper

<b>Text-Driven Tumor Synthesis</b> <br/>
[Xinran Li](https://scholar.google.com/citations?hl=zh-CN&user=awRZX_gAAAAJ)<sup>1,2</sup>, [Yi Shuai](https://openreview.net/profile?id=~Yi_Shuai1)<sup>3,4</sup>, [Chen Liu](https://scholar.google.com/citations?user=i938yiEAAAAJ&hl=zh-CN)<sup>1,5</sup>, [Qi Chen](https://scholar.google.com/citations?user=4Q5gs2MAAAAJ&hl=en)<sup>1,6</sup>, [Qilong Wu](https://github.com/JerryWu-code)<sup>1,7</sup>, [Pengfei Guo](https://scholar.google.co.uk/citations?hl=en&pli=1&user=_IAp-bYAAAAJ)<sup>8</sup>, [Dong Yang](https://scholar.google.com/citations?user=PHvliUgAAAAJ&hl=en&oi=sra)<sup>8</sup>, [Can Zhao](https://scholar.google.com/citations?user=CdzhxtYAAAAJ&hl=en)<sup>8</sup>, [Pedro R. A. S. Bassi](https://scholar.google.com/citations?hl=zh-CN&user=NftgL6gAAAAJ)<sup>1,9,10</sup>, [Daguang Xu](https://research.nvidia.com/person/daguang-xu)<sup>8</sup>, [Kang Wang](https://radiology.ucsf.edu/people/kang-wang)<sup>11</sup>, [Yang Yang](https://scholar.google.com/citations?user=6XsJUBIAAAAJ&hl=zh-CN)<sup>11</sup>, [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>1</sup>, [Zongwei Zhou](https://www.zongweiz.com/)<sup>1,*</sup> <br/>
<sup>1</sup> Johns Hopkins University  
<sup>2</sup> Shenzhen Technology University  <sup>3</sup> The First Affiliated Hospital of Sun Yat-sen University   
<sup>4</sup> Sun Yat-sen University  <sup>5</sup> Hong Kong Polytechnic University  
<sup>6</sup> University of Chinese Academy of Sciences  <sup>7</sup> National University of Singapore  
<sup>8</sup> NVIDIA  <sup>9</sup> University of Bologna  <sup>10</sup> Italian Institute of Technology  <sup>11</sup> University of California, San Francisco <br/>
<a href='https://arxiv.org/pdf/2412.18589'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a>

**We collect paper related to medical data synthesis in [Awesome Synthetic Tumors](https://github.com/MrGiovanni/SyntheticTumors/blob/main/AWESOME.md) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)**


## 0. Installation

```bash
git clone https://github.com/MrGiovanni/TextoMorph.git
cd TextoMorph
```
See [installation](utils/INSTALL.md) to obtain requirements and download dataset.


## 1. Train Diffusion Model

#### 📥 Download Required Files
We offer the pre-trained checkpoint of the **Autoencoder Model**, which was trained on the **AbdomenAtlas 1.1 dataset** (see details in [SuPreM](https://github.com/MrGiovanni/SuPreM)).  
This checkpoint can be directly used for the Diffusion model if you do not want to re-train the Autoencoder Model. Simply download it to `Diffusion/pretrained_models/` by running the following command:

```bash
cd Diffusion/pretrained_models
wget https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/AutoencoderModel/AutoencoderModel.ckpt
cd ../..
```
#### 💡 How to Train Your Own Model

Due to licensing constraints, we are unable to provide the training CT datasets. However, to assist in training your own model, we have made the **descriptive words** used during training available in the following files:

- [📁 liver](https://github.com/MrGiovanni/TextoMorph/tree/main/Diffusion/cross_eval/liver/real_tumor.txt)
- [📁 pancreas](https://github.com/MrGiovanni/TextoMorph/tree/main/Diffusion/cross_eval/pancreas/real_tumor.txt)
- [📁 kidney](https://github.com/MrGiovanni/TextoMorph/tree/main/Diffusion/cross_eval/kidney/real_tumor.txt)

If you wish to train your own model, you can rewrite these `real_tumor.txt` files using the following format:

```plaintext
CT_id  Label_id  t1  t2  ...  t100
```
#### 🔧 Start Training
```bash
cd Diffusion/
vqgan_ckpt="pretrained_models/AutoencoderModel.ckpt" # your-datapath
datapath="/ccvl/net/ccvl15/xinran/CT/" # your-datapath
tumorlabel="/ccvl/net/ccvl15/xinran/Tumor/liver/" # your-datapath
python train.py dataset.name=liver_tumor dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['liver'] dataset.uniform_sample=False model.results_folder_postfix="liver"  model.vqgan_ckpt=$vqgan_ckpt
```

We offer the pre-trained checkpoints of Diffusion Model for liver, pancreas and kidney, respectively. This checkpoint can be directly used for segmentation if you do not want to re-train the Diffusion Model. Simply download it to `Segmentation/TumorGeneration/model_weight`

#### 🔗 Checkpoints Overview

The following checkpoints are available for download. These pre-trained weights can be used directly for tumor generation and segmentation tasks for different organs:

| Tumor      | Download                                                                                      |
|------------|-----------------------------------------------------------------------------------------------|
| **Liver**  | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/liver.pt) |
| **Pancreas** | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/pancreas.pt) |
| **Kidney**   | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/kidney.pt?download=true) |

## 2. Train Segmentation Model

To use the **TumorGeneration** segmentation model, download the necessary pre-trained weights.  
Follow the instructions below to ensure the proper setup of your directory structure and model files.

#### 📥 Download Required Files

Run the following commands to download the pre-trained weights:

```bash

cd Segmentation/TumorGeneration/model_weight
wget https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/AutoencoderModel/AutoencoderModel.ckpt

wget https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/liver.pt
wget https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/pancreas.pt
wget https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/descriptivetumor2/kidney.pt

cd ../..
```
#### 🔧 Start Training
```bash
cd Segmentation

healthy_datapath="/ccvl/net/ccvl15/xinran/" # your-datapath
datapath="/ccvl/net/ccvl15/xinran/" # your-datapath
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
#### 🔗 Checkpoint Overview

| **Model**           | **Download**                                                                                 |
|----------------------|---------------------------------------------------------------------------------------------------|
| **Liver**     | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/segmentation/liver.pt) |
| **Pancreas**   | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/segmentation/pancreas.pt) |
| **Kidney**     | [Download](https://huggingface.co/Alena-Xinran/DescriptiveTumor/resolve/main/segmentation/kidney.pt) |

## 3. Evaluation


```bash
cd Segmentation
datapath="/ccvl/net/ccvl15/xinran/" #your-datapath
organ=liver
fold=0
datafold_dir=cross_eval/"$organ"_aug_data_fold/
python -W ignore validation.py --model=unet --data_root $datapath --datafold_dir $datafold_dir --tumor_type tumor --organ_type $organ --fold $fold --log_dir $organ/$organ.fold$fold.unet --save_dir out/$organ/$organ.fold$fold.unet
```

## Citation

```
@article{li2024text,
  title={Text-Driven Tumor Synthesis},
  author={Li, Xinran and Shuai, Yi and Liu, Chen and Chen, Qi and Wu, Qilong and Guo, Pengfei and Yang, Dong and Zhao, Can and Bassi, Pedro RAS and Xu, Daguang and others},
  journal={arXiv preprint arXiv:2412.18589},
  year={2024},
  url={https://github.com/MrGiovanni/TextoMorph}
}

@article{chen2024analyzing,
  title={Analyzing Tumors by Synthesis},
  author={Chen, Qi and Lai, Yuxiang and Chen, Xiaoxi and Hu, Qixin and Yuille, Alan and Zhou, Zongwei},
  journal={arXiv preprint arXiv:2409.06035},
  year={2024}
}

@inproceedings{chen2024towards,
  title={Towards Generalizable Tumor Synthesis},
  author={Chen, Qi and Chen, Xiaoxi and Song, Haorui and Xiong, Zhiwei and Yuille, Alan and Wei, Chen and Zhou, Zongwei},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024},
  url={https://github.com/MrGiovanni/DiffTumor}
}
```

## Acknowledgments
This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the Patrick J. McGovern Foundation Award.
