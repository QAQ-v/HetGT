# HetGT
An implementation for paper "Heterogeneous Graph Transformer for Graph-to-Sequence Learning" (accepted at ACL20).

## Requirements
```pip install -r requirements.txt```

## Preprocessing
Transform the input graphs into Levi graphs. The code is modified from Konstas et al. (2017) and Guo et al., 2019.

For AMR-to-text, get the AMR Sembank (LDC2017T10) first and put the folder called abstract_meaning_representation_amr_2.0 inside the data folder. Then run:

```./gen_amr.sh```

For NMT, you can download the raw dataset from here first and change the data folder inside nmt_preprocess.py. Then run:

```python nmt_preprocess.py```

Then we introduce BPE into Levi graph, here we take en2cs as the example, please ensure that the path of en2cs folder exist (data/nmt19/en2cs):

```bash preprocess.bpe.sh 8000 cs```

Then we will get the folder called data/nmt19/en2cs.bpe.8000.both.new. Prepare data:

```bash preprocess.sh data/nmt19/en2cs.bpe.8000.both.new ```

## Training

``` bash train.sh data/nmt19/en2cs.bpe.8000.both.new/amr2015 model/nmt19/cs/model 0 log/cs jump ```


## Prediction and Evaluation

```bash predict.sh model/nmt19/cs/model/ADAM_acc61.23_ppl8.80_lr0.00021_step170000.pt data/nmt19/en2cs.bpe.8000.both.new 0 cs ```

## Citation
