# Prepare data
Files in this directory are used to prepare data for MRML with meta-data in Weibo and Twitter datasets.
The gen_by_VGG.py file is for generating visual unimodal representations of images in Weibo and Twitter datasets. 
The gen_emb_Bert.py file is used to generate textual unimodal representations of text content in Weibo and Twitter datasets and prepare the representation files for MRML.

# Note
The environment for keras-bert is different from the MRML. 
The python version is python-3.8.6 and the required version of packages are in requirements.txt.
You can install all the required packages using the following command:
```
$ pip install -r requirements.txt
```
