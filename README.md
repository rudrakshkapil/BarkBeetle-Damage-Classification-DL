# Classification of Bark Beetle-Induced Forest Tree Mortality using Deep Learning

This is the code repository for the extended abstract submitted to the [VAIB workshop](https://homepages.inf.ed.ac.uk/rbf/vaib22.html) to be presented at ICPR 2022. The [pre-print](https://arxiv.org/abs/2207.07241) is available on ArXiv. The task is to classify individual tree crowns into four different attack stages of bark beetle infestation from UAV-captured RGB images. 

![Classification task summarized.](https://github.com/rudrakshkapil/BarkBeetle-Damage-Classification-DL/blob/main/vaib%20block_diagram.png?raw=true)



## Repository Organization
The python scripts in this repository are for prcoessing the data, training the model, and evaluating the performance. As noted in the extended abstract, we made modifications to a RetinaNet-based architecture (i.e. [DeepForest](https://deepforest.readthedocs.io/en/latest/landing.html)). The modifications to this package are found in `/df_repo`. 

`/plots` contains some visualizations of the dataset and performance of the model.


## Obtaining Dataset and Trained Models
These are too large to host on GitHub, so they are instead available from [this Google Drive link](https://drive.google.com/drive/folders/1-TisGZ9vo5hqp-0aawMW7IeVaW50CMWG?usp=sharing). There is one trained model for each flight -- all are from the experiment that yielded the best average performance. 

The dataset is arranged into multiple folders. The images in `datasets/cropped_squares` are tree crown collections obtained from the orthomosaics in `datasets/mosaics` for a specific flight using the provided bounding box information in `datasets/annotations`. The flight is specified during training and evaluation in `train.py`, and the cropped squares folder will contain only those images.

The `datasets` directory needs to be downloaded and placed into this repository for the scripts to function correctly. 

## Installation and Training
The python packages needed to run this code are specified in `environment.yml`. You can create a new conda environment from this file. Enter into a terminal:

`conda env create -f environment.yml`
`conda activate bb_damage`

The environment name is the first line of environment.yml, i.e. *bb_damage*.

If any packages are missing or incorrectly installed, you can install just those specific ones later on. Make sure the *bb_damage* environment is activated, then type:

`conda install <PACKAGE_NAME>`

Note: There may be an incompatibility issue with the provided `environment.yml` file and seaborn (which is needed for one of the plots in `visualize.py`). If you need to run this script, we recommend creating and using another conda environment with only the required pacakges for seaborn for running this script. 

## Citation
If you find this repository useful for your research, please cite the [pre-print on ArXiv](https://arxiv.org/abs/2207.07241). 
