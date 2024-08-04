## Pytorch implementation of [MGC](https://ieeexplore.ieee.org/document/10553237)

![image](https://github.com/benesakitam/MGC/blob/main/figs/pipeline.png)

## Requirements

To install requirements:

```setup
conda create -n mgc python=3.7
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
pip install tensorboardX
pip install ipdb
pip install einops
pip install loguru
pip install pyarrow==3.0.0
pip install imdb
pip install tqdm
```

## Data Preparation
Download scene classification datasets: AID, MLRSNet, and NWPU45.

If your folder structure is different, you may need to change the data structures as follows.
``` data structure
├── AID
│   ├── Airport
│   │   ├── airport_1.jpg
│   │   ├── airport_2.jpg
│   │   ├── ...
│   ├── BareLand
│   │   ├── bareland_1.jpg
│   │   ├── bareland_1.jpg
│   │   ├── ...
│   ├── ...

├── MLRSNet
│   ├── airplane
│   │   ├── airplane_00001.jpg
│   │   ├── airplane_00002.jpg
│   │   ├── ...
│   ├── airport
│   │   ├── airport_00001.jpg
│   │   ├── airport_00002.jpg
│   │   ├── ...
│   ├── ...

├── NWPU45
│   ├── airplane
│   │   ├── airplane_001.jpg
│   │   ├── airplane_002.jpg
│   │   ├── ...
│   ├── airport
│   │   ├── airport_001.jpg
│   │   ├── airport_002.jpg
│   │   ├── ...
│   ├── ...
```

Generate the Fusion dataset. Replace the original data path in MGC/tools/fusion_generation.py (Line 9, 14, and 19) with your {data_path}.
```Fusion generation
python tools/fusion_generation.py
```
Spilt AID, MLRSNet, and NWPU45, respectively. Replace the original data path in MGC/tools/split_dataset.py (Line 34, 35, and 36) with your {data_path}.
```split dataset
python tools/split_dataset.py
```
Replace the original data path in MGC/data/folder2lmdb.py (Line 137). Note generate only the lmbd file for the  training set of Fusion.
```generate lmdb
python data/folder2lmdb.py
```
Relpace the original data path in MGC/data/dataset_lmdb.py (Line7 and Line40) with your pre-training or evaluation {data_path}.

## Pre-Training

Before pre-training default MGC-B, run this command first to add your PYTHONPATH:

```train
export PYTHONPATH=$PYTHONPATH:{your_code_path}/MGC/
```

Then run the training code via:

```train
python -m torch.distributed.launch --nproc_per_node 4 tools/train_new.py -b 512 -n 2 -d 0-3 --experiment-name mgc-b --distributed --world-size 1
```

You can pre-train MGC-L model by modifying the models/mlps.py (Line 149-154).

>📋  The pre-training command is used to do unsupervised pre-training of a ResNet-50 model on Fusion in an 4-gpu machine
>1. using `-b` to specify batch_size, e.g., `-b 512`
>2. using `-d` to specify gpu_id for training, e.g., `-d 0-3`
>3. using `--log_path`  to specify the main folder for saving experimental results.
>4. using `--experiment-name` to specify the folder for saving training outputs.
>
## Evaluation
Before start the evaluation, run this command first to add your PYTHONPATH:

```eval
export PYTHONPATH=$PYTHONPATH:{your_code_path}/MGC/
```

Select output num_classes of the classifer in exps/linear_eval_exp.py (Line 17, 19, 21) when you evaluate different datasets.

Then, to evaluate the pre-trained model on AID, MLRSNet, and NWPU45 respectively, run:
```eval
python -m torch.distributed.launch --nproc_per_node 4 tools/eval_new.py -b 512 -ckpt 800 --experiment-name mgc-b --distributed --world-size 1
```

>📋  The evaluation command is used to do the supervised linear evaluation of a ResNet-50 model on AID, MLRSNet, and NWPU45 in an 4-gpu machine, respectively.
>1. using `-b` to specify batch_size, e.g., `-b 512`
>2. using `-d` to specify gpu_id for training, e.g., `-d 0-3`
>3. Modifying `--log_path`  according to your own config.
>4. Modifying `--experiment-name` according to your own config.
>5. Don't forget to modify ```data/folder2lmdb.py``` to ensure the path to the evaluation dataset is correct.

    
## Other downstream tasks
Convert the pre-trained MGC model to unique format.
```eval
python tools/convert_model.py --input results/train_new_log/mgc-b/800_ckpt.pth.tar --output {your_path}/mgc-b_pre-trained.pth
```
For other downstream tasks including Rotated Object Detection, Semantic Segmentation, and Change Detection. Please refer (https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing).

Replace the pre-trained model path in the config files of each downstream tasks with the path of our converted MGC models.

## Visualization

![image](https://github.com/benesakitam/MGC/blob/main/figs/vis.jpg)

## Citation
Please cite this paper if it helps your research:
```
@article{li2024mgc,
  title={MGC: MLP-Guided CNN Pre-Training using A Small-scale Dataset for Remote Sensing Images},
  author={Li, Zhihao and Hou, Biao and Li, Wanqing and Wu, Zitong and Ren, Bo and Jiao, Licheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
