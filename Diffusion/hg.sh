

vqgan_ckpt=pretrained_models/AutoencoderModel.ckpt
# datapath="/ccvl/net/ccvl15/xinran/CT/"
# tumorlabel="/ccvl/net/ccvl15/xinran/Tumor/liver/"
# datapath="/ccvl/net/ccvl15/xinran/CT/"
# tumorlabel="/ccvl/net/ccvl15/xinran/Tumor/pancreas/"
datapath="/ccvl/net/ccvl15/xinran/CT/"
tumorlabel="/ccvl/net/ccvl15/xinran/Tumor/kidney/"
#/mnt/ccvl15/cliu234/miniconda3/envs/difftumor/bin/python train.py dataset.name=liver_tumor_train dataset.fold=$fold dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['liver_tumor_data_early_fold'] dataset.uniform_sample=False model.results_folder_postfix="liver_early_tumor_fold0" model.vqgan_ckpt=$vqgan_ckpt
python3 train.py dataset.name=kidney_tumor dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['kidney'] dataset.uniform_sample=False model.results_folder_postfix="kidney" model.vqgan_ckpt=$vqgan_ckpt

# python3 train.py dataset.name=liver_tumor dataset.fold=$fold dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['liver_tumor_data_early_fold'] dataset.uniform_sample=False model.results_folder_postfix="liver_early_tumor_fold0" model.vqgan_ckpt=$vqgan_ckpt

# python3 train.py dataset.name=pancreas_tumor dataset.fold=$fold dataset.data_root_path=$datapath dataset.label_root_path=$tumorlabel dataset.dataset_list=['pancreas_tumor_data_early_fold'] dataset.uniform_sample=False model.results_folder_postfix="pancreas_early_tumor_fold0" model.vqgan_ckpt=$vqgan_ckpt

# sbatch --error=logs/diffusion_model.out --output=logs/diffusion_model.out hg.sh
