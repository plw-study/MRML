# Paper for ICASSP 2023
Pytorch code for paper: "MRML: Multimodal Rumor Detection by Deep Metric Learning"

# Overview
This directory contains code necessary to run the MRML. MRML is a multimodal rumor detection network by deep metric learning. See our paper for details on the code.

# Dataset
The meta-data of the Weibo and Twitter datasets used in our experiments are available in their papers. 
In this project, we provide the py files for data preparing in the pre_data subdirectory. 

# Requirements
The python version is python-3.6.4. The detailed version of some packages is available in requirements.txt.
You can install all the required packages using the following command:
```
$ pip install -r requirements.txt
```

# Running the code
The train.py is the main file for running the code.
```
$ python train.py
```
# Reference
Detailed data analysis and method are in our paper:
```
@INPROCEEDINGS{peng-MRML,
  author={Peng, Liwen and Jian, Songlei and Li, Dongsheng and Shen, Siqi},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={MRML: Multimodal Rumor Detection by Deep Metric Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096188}}

```
