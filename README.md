# Measure Visual Commonsense Knowledge

<img src="images/overview.png" width="400"/>

## ACL SRW 2022 paper

["What do Models Learn From Training on More Than Text? Measuring Visual Commonsense Knowledge"](https://arxiv.org/abs/2205.07065). 

This repo contains the code for the paper.

&nbsp;

## Where to start?

The repo is segmented into three main parts:

1. [models](models) contains code necessary for attaining the models that haven't already been pre-trained and released. These are the BERT baselines trained on visual copora (`bert-clip-bert-train`, `bert-lxmert-train` and `bert-lxmert-train-scratch`) and CLIP-BERT. This repo also contains necessary model weights and code for pretraining.
2. [memory_colors](memory_colors) contains code necessary for the Memory Colors evaluation. As long as you have the necessary model weights under `models/data/model-weights`, this can be run independently from the other directories.
3. [visual_property_norms](visual_property_norms) contains code necessary for the Visual Property Norms evaluation. As long as you have the necessary model weights under `models/data/model-weights`, this can be run independently from the other directories.

Both the Memory Colors evaluation and the Visual Property Norms evaluation depend on pre-trained model weights for the models evaluated. Some of this pre-training needs to be done separately in [models](models).

## Acknowledgements
This project wouldn't be possible without the Centre for Speech, Language, and the Brain (CSLB) at the University of Cambridge, the [Huggingface](https://huggingface.co/) library and the [LXMERT repo](https://github.com/airsplay/lxmert), we thank you for your work!
