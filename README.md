# CSL
This repository contains the reference code for the a novel video-text retrieval method "Complementarity-aware Space Learning", codes and trained models will be released soon!

## Environment setup
* Python 3.6
* Pytorch 1.7.1 (strongly recommand)
* Numpy 1.19

## Data preparation
1. Please follow [dual_enc](https://github.com/danieljf24/hybrid_space) to download the required features for MSRVTT.
Then put the folder “msrvtt10k” under ${DataPath }.

2. Please follow [HGR](https://github.com/cshizhe/hgr_v2t) to download the required annotations (int2word.json, word2int.json, word_embeds.glove32b.th).
Then put them under the file ${DataPath }.

```bash
# clone the repository
git clone 
cd CSL
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

## Inference procedure
To reproduce the results of our paper,  do the following two steps:
1. modify the /path/to/data in ./inference.py into $DataPath 

2. please run the code below:
```bash
python inference.py
```
## Training procedure
To train a our CSL model, do the following two steps:
1. modify the /path/to/data in ./train_msrvtt.py into $DataPath
2. please run the code below:
```bash
cd ./
python main.py
```
Note that it takes 1 v100 GPUs and around 20 hours to train this model.
