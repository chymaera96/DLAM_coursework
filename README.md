# DLAM_coursework

This repository contains the network architecture for training, extracting fingerprints and inferencing genre classification for test dataset. The embedding extraction and the inferencing (classification model) is compatible with "Contrastive Learning for Musical Representatation (CLMR)" pretrained-weights and it is used as a baseline for the classification task.  For detailed demonstration of the performance of the models, look at the colab notebook  `DLAM_demo.ipynb`. 

## Installation Notes

* The `requirements.txt` all relevant packages.
* Refer to the colab notebook for CLMR embedding extraction.


## Other functionalities

Most of the requisite information for validation of performance is demonstrated in the colab notebook. However, here are some additional information on the pipelines:

* To run contrastive training of the Slow-Fast Learning encoder
```
python train.py --data_dir=path/to/train/data --ir_data=path/to/impulse/response/data --noise_dir=path/to/background/noise/data
```

* To run fingerprinting generation and compute query matching hit rates
```
python extract_embeddings.py --test_dir=path/to/test/data --ckp=path/to/checkpoint
python test_fp.py --test_dir=path/to/test/data --ckp=path/to/checkpoint
```
