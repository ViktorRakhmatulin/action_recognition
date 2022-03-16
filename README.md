# Real-Time Action Detection and Prediction in Human Motion Streams

<p align=center>
<img src="images/3144.gif" width="50%"><img src="images/3292.gif" width="50%">
</p>

This code runs experiments for a real-time action detection in motion capture data implemented with LSTMs.
It reproduces experiments presented in the following paper:
```
Carrara, F., Elias, P., Sedmidubsky, J., & Zezula, P. (2019).
LSTM-based real-time action detection and prediction in human motion streams.
Multimedia Tools and Applications, 78(19), 27309-27331.
```
Experiments are conducted on the HDM-05 dataset. _**NOTE**: Few sequences from the HDM05 dataset are partially missing labels.
The above videos show two sequences of this kind. The prediction of our model is on top, while the (wrong) groundtruth is on the bottom._

## How to reproduce (Comments from Viktor)

Configured via Anaconda-Navigator.  
Tested on Ubuntu 20.04. Python 3.6. 
Executed  the following command in conda environment (for cuda support)

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
taken from https://pytorch.org/


1. Download the preprocessed data archive and extract it in the repo root folder: [hdm05-mocap-data.tar.gz](https://drive.google.com/file/d/1YyQTS2vyK0Z6MdeTd9ko9K3u_E6G8i5c/view?usp=sharing) (~1GB, the original HDM05 dataset is available [here](http://resources.mpi-inf.mpg.de/HDM05/))

2. Run 'bash parse_HDM05_data.sh` to generate data splits 
3. Run 'bash train_classification_models.sh` to train an LSTM-based classification model 
Segmentation as far as I understood is aimed to find the beginning and the end of the action in given sequence.
We are more interested in classification (mapping a sequence to a class) 
Python train_classify.py`  
4.  `train_segmentation_models.sh` contains examples for SEGMENTATION task

