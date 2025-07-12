### Enviroment
To get te correct enviroment to run the code, it must follow the next commands:
```
conda create -n maskformer python==3.9
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install -r requirements.txt
pip install setuptools==59.5.0
```

### Dataset
The data must be save with the next structure:
--Fold_1
----train
------images
------masks
----val
------images
------masks

or can be found in the folder: ../shared data/maskformer_data/colorectal_fold

In the train_net.py file must be update the folder_data in the main function

### Training 
To train the model is necessary to download the baseline checkpoint in [MaskFormer Model Zoo](MODEL_ZOO.md) of the choosen model. To use this checkpoint in the model the config model it must be modified, for the case of swin base (the used model) the file [MaskFormer swin base config file](configs/ade20k-150/swin/maskformer_swin_base_IN21k_384_bs16_160k_res640.yaml) must be updated in **WEIGHTS**

The next command is for training :
```
python train_net.py configs/ade20k-150/swin/maskformer_swin_base_IN21k_384_bs16_160k_res640.yaml
```
