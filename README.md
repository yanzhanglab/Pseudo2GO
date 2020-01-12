# Pseudo2GO

## Description
This is a graph-based deep learning method for predicting pseudogene functions by borrowing information from coding genes. We use both network information and node attributes to improve the performance. Sequence similarity networks are used to construct graphs connecting pseudogenes and coding genes, which are used to propagate node attribtues, so that pseudogenes can borrow information from well-studied coding genes.

We use two types of expression profiles (from TCGA and GTEx database, respectively), interactions with microRNAs and PPI and genetic interactions as the node attributes (initial feature representation).

We have shown that our method achieved state-of-the-art performance, significantly outperforming existing methods. Our graph neural network model is implemented based on Pytorch Geometric package in Python 3.6.

## Usage
### Requirements
- Python 3.6
- Pytorch
- Pytorch Geometric
- networkx
- scipy
- numpy
- pickle
- scikit-learn
- pandas

### Data
You can download the raw data and processed data (ready for use in the model) from here <a href="https://www.dropbox.com/sh/7hamubz2dgityrs/AAAqiSDh8XRWdFjqJU-EaK6ja?dl=0" target="_blank">data</a>. Please Download the datasets and put them in the existing data folder.

### Steps
#### Step1: decompress data files
> unzip data.zip       
> unzip raw_data.zip        
> unzip final_input.zip       
> mv raw_data final_input data

#### Step2: preprossing (Optional)
> cd preprocessing          
> python preprocess_final.py

#### Step2: run the model
> cd model    
> python pseudo2go.py    
> **Note there are several parameters can be tuned. Please refer to the pseudo2go.py file for detailed description of all parameters**
