# Taxonomy of hybridly polarized Stokes vortex beams

Paper Link: https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-5-7404&id=546969

## Citation

Please consider citing the below paper, if you are using the code provided in this repository.
```
@article{arora2024taxonomy,
  title={Taxonomy of hybridly polarized Stokes vortex beams},
  author={Arora, Gauri and Butola, Ankit and Rajput, Ruchi and Agarwal, Rohit and Agarwal, Krishna and Horsch, Alexander and Prasad, Dilip K and Senthilkumaran, Paramasivam},
  journal={Optics Express},
  volume={32},
  number={5},
  pages={7404--7416},
  year={2024},
  publisher={Optica Publishing Group}
}
```

## Overview
This repository contains implementation codes of all the deep learning models used in the [above paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-5-7404&id=546969).

## Design
In our paper, we use three below-written experimental designs. For more details, please refer to the paper.
1. **Design 1 -** *Training with only simulation data*: We trained and validated all five deep-learning models using simulated data and tested on the experimental data. 
2. **Design 2 -** *10-fold strategy on experimental data*: We employed a 10-fold strategy to train a ResNet-18 model.
3. **Design 3 -** *Mix Training*:  We train the deep-learning models on mix simulated and experimental data for generalized and robust training.

## Datasets

The data that support the findings of this study are available from the corresponding author upon reasonable request. All the links below are private. Upon request, we can provide you access to this link.

For information on the datasets, please refer to the [paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-5-7404&id=546969).

The datasets needs to be stored in the `data` folder. Download the data from the below links and save inside the `data` folder.
1. Design 1: https://figshare.com/s/ecb99ffe720944d031e4
2. Design 2: https://figshare.com/s/a21f0f14372a634619c4
3. Design 3: https://figshare.com/s/b50f38c5612477c5af95

## Dependencies

## Running the code
To run the models, change the control parameters accordingly in the **main.py** file and run
```
python Code/main.py --design 1
```

### Control Parameters
- `--design`: The type of experiment design. \
          choices = [1, 2, 3] \
          default=3
- `--model_name`: Model to employ \
          choices=["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"] \
          default="densenet"
- `--use_pretrained`: Use Pretrained model on ImageNet \
          default="True"
- `--feature_extract`: Flag for feature extracting. When False, we finetune the whole model otherwise we only update the reshaped layer params \
          default="False"
- `--batch_size`: Batch size for training (change depending on how much memory you have).
- `--num_epochs`: Number of epochs to train for.
- `--lr`: Learning rate of the model.
- `--momentum`: momentum of the model.
- `--pretrain_simulated`: If design is 2, then this is a must variable. It is true in case the model need to be pretrain on simulated data otherwise False.

**Note: If design = 2, then make sure to assign `pretrain_simulated` variable**

## Results
The results are saved in the results folder in `.data` format. Please read them to get your prediction results. 

`AnalyseResult` folder contains codes to analyse results. Provide correct result paths to run the code.

