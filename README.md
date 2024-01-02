# Paper for ICASSP 2023
Pytorch code for paper: "MRML: Multimodal Rumor Detection by Deep Metric Learning"

# Overview
This directory contains code necessary to run the MRML. MRML is a multimodal rumor detection network by deep metric learning. See our paper for details on the code.

# Dataset
The meta-data of the Weibo and Twitter datasets used in our experiments are available in their papers. 

- Multimodal Fusion with Recurrent Neural Networks for Rumor Detection on Microblogs [论文地址](https://dl.acm.org/doi/10.1145/3123266.3123454)

- Verifying Multimedia Use at MediaEval 2016 [项目地址](https://github.com/MKLab-ITI/image-verification-corpus/tree/master/mediaeval2016)

In this project, we provide the py files for data preparing in the pre_data subdirectory. 
The meta-data can be downloaded in the following:

- weibo[下载地址](https://pan.baidu.com/s/1S0OxCWRvXsP2cOWdDt_BRg), 提取码：4j7p

- twitter[下载地址](https://pan.baidu.com/s/1nQEJTtY2Dm8Jdrn4r5ofRQ)， 提取码：vc8h
  

# Requirements
It is recommended to create an anaconda virtual environment to run the code.
The python version is python-3.6.4. The detailed version of some packages is available in requirements.txt.
You can install all the required packages using the following command:
```
$ conda install --yes --file requirements.txt
```

# Running the code
The train.py is the main file for running the code.
```
$ python train.py
```
# Reference
Detailed data analysis and method are in our paper.
If you are insterested in this work, and want to use the dataset or codes in this repository, please star this repository and cite by:
```
@INPROCEEDINGS{peng-MRML,
  author={Peng, Liwen and Jian, Songlei and Li, Dongsheng and Shen, Siqi},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={MRML: Multimodal Rumor Detection by Deep Metric Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096188}
}
```
