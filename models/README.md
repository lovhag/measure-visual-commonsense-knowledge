# Models

This folder contains code necessary for running and training the models evaluated in this work. The models we evaluate are:

* BERT-base
* CLIP-BERT
* LXMERT
* VisualBERT

## Model code

* BERT-base code is provided by the Huggingface library
* CLIP-BERT code can be found under `src/clip_bert`
* Code for altering LXMERT to work with unimodal language input can be found under `src/lxmert`
* VisualBERT code is provided by the Huggingface library

## Model training

The following models were further trained by us for the purposes of this work:
* CLIP-BERT, 
* BERT-CLIP-BERT-train, 
* BERT-LXMERT-train, and 
* BERT-LXMERT-train-scratch

The specifics for how this was performed can be found in this document.

## Model weights

To obtain the trained model weights, you can run the training scripts below. Or you can [download the model weights](https://chalmers-my.sharepoint.com/:f:/g/personal/lovhag_chalmers_se/ErO_VM7bO5FHlq0EkVJzLMsBxVwysfIjGPyqf-pqOU9LoQ?e=Ew288k) and put them under `models/data/model-weights`.

### CLIP-BERT

This section describes the experiment of training a CLIP-BERT model from pre-trained BERT weights.

Firstly, download the images and corpora for Conceptual Captions, SBU Captions, COCO and Visual Genome QA. Then, create the training data files `models/data/clip-bert/VLP/train.jsonl`, `models/data/clip-bert/VLP/val.jsonl` and `models/data/clip-bert/VLP/clip_features.hdf5` using the makefile `models/data/clip-bert/VLP.mk`. 

The model can then be trained using the following command:

```
models/data/runs/clip-bert/run.sh
```

We trained the model on four NVIDIA Tesla T4 GPU with 16GB RAM and 8 cores for 16h. The model appeared to have converged at this time.

### BERT-CLIP-BERT-train

This section describes the experiment of training a BERT base uncased model on the language part of the CLIP-BERT training data from pre-trained weights.

See the CLIP-BERT section above for how to acquire the necessary CLIP-BERT data.

The model can then be trained using the following command:

```
models/data/runs/bert-clip-bert-train/run.sh
```

We trained the model on four NVIDIA Tesla T4 GPU with 16GB RAM and 8 cores for 16h. The model appeared to have converged at this time.

### BERT-LXMERT-train-scratch

This section describes the experiment of training a BERT base uncased model on the language part of the LXMERT training data from scratch.

The LXMERT pre-training data can be downloaded via https://github.com/airsplay/lxmert#pre-training, put it under `models/data/lxmert`. Then, format the data by running the [process_train_data_for_MLM.ipynb](models/src/lxmert/process_train_data_for_MLM.ipynb) notebook. As a result you should get the files `models/data/lxmert/train_mlm.jsonl` and `models/data/lxmert/val_mlm.jsonl`.

The model can then be trained using the following command:

```
models/data/runs/bert-lxmert-train-scratch/run.sh
```

We trained the model on four NVIDIA Tesla T4 GPU with 16GB RAM and 8 cores for 9h. The model appeared to have converged at this time.

### BERT-LXMERT-train

This section describes the experiment of training a BERT base uncased model on the language part of the LXMERT training data from pre-trained BERT weights.

See the BERT-LXMERT-train-scratch section above for how to acquire the necessary LXMERT data.

The model can then be trained using the following command:

```
models/data/runs/bert-lxmert-train/run.sh
```

We trained the model on four NVIDIA Tesla T4 GPU with 16GB RAM and 8 cores for 9h. The model appeared to have converged at this time.