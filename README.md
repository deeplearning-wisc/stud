# STUD

This is the source code accompanying the paper ***Unknown-Aware Object Detection:Learning What You Don’t Know from Videos in the Wild*** [paper](https://arxiv.org/abs/2203.03800) by Xuefeng Du, Xin Wang, Gabriel Gozum and Yixuan Li 

The codebase is based heavily from [CycleConf](https://github.com/xinw1012/cycle-confusion) and [Detectron2](https://github.com/facebookresearch/detectron2).

## Ads 

Checkout our
* ICLR'22 work [VOS](https://github.com/deeplearning-wisc/vos) on object detection in still images and classification networks.
* NeurIPS'22 work [SIREN](https://github.com/deeplearning-wisc/siren) on OOD detection for detection transformers.
* ICLR'23 work [NPOS](https://openreview.net/forum?id=JHklpEZqduQ) on non-parametric outlier synthesis.
* NeurIPS'23 work [DREAM-OOD](https://arxiv.org/pdf/2309.13415.pdf) on outlier generation in the pixel space (by diffusion models) if you are interested!

## Installation

### Environment
- CUDA 10.2
- Python >= 3.7
- Pytorch >= 1.6
- THe Detectron2 version matches Pytorch and CUDA versions.

### Dependencies

1. Create a virtual env.
- `python3 -m pip install --user virtualenv`
- `python3 -m venv stud`
- `source stud/bin/activate`

2. Install dependencies.

- `pip install -r requirements.txt`

- Install Pytorch 1.9

`pip3 install torch torchvision`

Check out the previous Pytorch versions [here](https://pytorch.org/get-started/previous-versions/).

- Install Detectron2
Build Detectron2 from Source (gcc & g++ >= 5.4)
`python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

Or, you can install Pre-built detectron2 (example for CUDA 10.2, Pytorch 1.9)

`python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html`

More details can be found [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).


## Data Preparation

**BDD100K**
1. Download the BDD100K MOT 2020 dataset (`MOT 2020 Images` and `MOT 2020 Labels`) and the detection labels (`Detection 2020 Labels`) [here](https://bdd-data.berkeley.edu/) and the detailed description is available [here](https://doc.bdd100k.com/download.html). Put the BDD100K data under `datasets/` in this repo. After downloading the data, the folder structure should be like below:
```
├── datasets
│   ├── bdd100k
│   │   ├── images
│   │   │    └── track
│   │   │        ├── train
│   │   │        ├── val
│   │   │        └── test
│   │   └── labels
│   │        ├── box_track_20
│   │        │   ├── train
│   │        │   └── val
│   │        └── det_20
│   │            ├── det_train.json
│   │            └── det_val.json
│   ├── waymo
```

Convert the labels of the MOT 2020 data (train & val sets) into COCO format by running:
```python
python3 datasets/bdd100k2coco.py -i datasets/bdd100k/labels/box_track_20/val/ -o datasets/bdd100k/labels/track/bdd100k_mot_val_coco.json -m track
python3 datasets/bdd100k2coco.py -i datasets/bdd100k/labels/box_track_20/train/ -o datasets/bdd100k/labels/track/bdd100k_mot_train_coco.json -m track
```

**COCO**

Download COCO2017 dataset from the [official website](https://cocodataset.org/#home). 

Download the OOD dataset (json file) when the in-distribution dataset is Youtube-VIS from [here](https://drive.google.com/file/d/1vLMGn7401-dEi5smxjgjr-IXhAXvjuf-/view?usp=sharing). 

Download the OOD dataset (json file) when the in-distribution dataset is BDD100k from [here](https://drive.google.com/file/d/1L4I7X-a3fojIJ9Y_NvT-SzieAabBARsW/view?usp=sharing).

Put the two processed OOD json files to ./anntoations

The COCO dataset folder should have the following structure:
<br>

     └── datasets
         └── coco2017
             ├── annotations
                ├── xxx (the original json files)
                ├── instances_val2017_ood_wrt_bdd.json
                └── instances_val2017_ood_wrt_vis.json
             ├── train2017
             └── val2017

**Youtube-VIS**

Download the dataset from the [official website](https://competitions.codalab.org/competitions/28988#participate-get_data).

Preprocess the dataset to generate the training and validation splits by running:
```python
python datasets/convert_vis_val.py
```

The Youtube-VIS dataset folder should have the following structure:
<br>

     └── datasets
        └── vis
          └── train
            └── JPEGImages
            ├── instances_train.json
            └── instances_val.json




**nuImages**

Download the dataset from the [official website](https://www.nuscenes.org/download).

Convert the dataset by running:
```python
python datasets/convert_nu.py
python datasets/convert_nu_ood.py
```

The nuImages dataset folder should have the following structure:
<br>

     └── datasets
        └── nuscence
          └── v1.0-mini
          ├── v1.0-test
          ├── v1.0-val
          ├── v1.0-train
          ├── samples
          ├── semantic_masks
          ├── calibrated
          ├── nuimages_v1.0-val.json
          └── nu_ood.json


Before training, modify the dataset address in the ./src/data/builtin.py according to your local dataset address.

## Training

**Vanilla with BDD100K as the in-distribution dataset**
```python
python -m tools.train_net --config-file ./configs/BDD100k/R50_FPN_all.yaml --num-gpus 4
```
**Vanilla with Youtube-VIS as the in-distribution dataset**
```python
python -m tools.train_net --config-file ./configs/VIS/R50_FPN_all.yaml --num-gpus 4
```
**STUD on ResNet (BDD as ID data)**
```python
python -m tools.train_net --config-file ./configs/BDD100k/stud_resnet.yaml --num-gpus 4
```
**STUD on RegNet (BDD as ID data)**
```python
python -m tools.train_net --config-file ./configs/BDD100k/stud_regnet.yaml --num-gpus 4
```

Download the pretrained backbone for RegNetX from [here](https://drive.google.com/file/d/1MjK9m68lAXj6AiVuSBMZX0m9DkOS2zRL/view?usp=sharing).

**Pretrained models**

The pretrained models for BDD100K can be downloaded from [vanilla](https://drive.google.com/file/d/19FSgMpzuRsl_qBZR4ifHq2soLDBmThFd/view?usp=sharing) and [STUD-ResNet](https://drive.google.com/file/d/1JAthrSJ8yK5bbhlZAD2vVor2uJ5aTkaX/view?usp=sharing) and [STUD-RegNet](https://drive.google.com/file/d/1-bqcdJjL3M8w09GRhRjuzawPEfX2SuUH/view?usp=sharing).

The pretrained models for Youtube-VIS can be downloaded from [vanilla](https://drive.google.com/file/d/1yKK9yDdLc_r2NSTaM5oYJ_09umi1GNwF/view?usp=sharing) and [STUD-ResNet](https://drive.google.com/file/d/1DTc2GqfNybcsFPnFA5o48E28MKCBTCFZ/view?usp=sharing) and [STUD-RegNet](https://drive.google.com/file/d/1hDdFYfWHl-SXUMd7xN9PcXnipkN-m5Qf/view?usp=sharing).

## Evaluation

**Evalutation with the in-distribution dataset to be BDD100K**

Firstly run on the in-distribution dataset:
```python
python -m tools.train_net --config-file ./configs/BDD100k/stud_resnet.yaml --num-gpus 4 --eval-only MODEL.WEIGHTS address/model_final.pth
```
where "address" is specified in the corresponding yaml file.

Then run on the OOD dataset (COCO):
```python
python -m tools.train_net --config-file ./configs/BDD100k/stud_resnet_ood_coco.yaml --num-gpus 4 --eval-only MODEL.WEIGHTS address/model_final.pth
```
Obtain the metrics using:
```python
python bdd_coco.py --energy 1 --model xxx
```
Here "--model" means the name of the directory that contains the checkpoint file. Evaluation on nuImages is similar.


## Citation ##
If you found any part of this code is useful in your research, please consider citing our paper:

```
  @article{du2022stud,
      title={Unknown-Aware Object Detection: Learning What You Don’t Know from Videos in the Wild}, 
      author={Du, Xuefeng and Wang, Xin and Gozum, Gabriel and Li, Yixuan},
      journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2022}
}
```
