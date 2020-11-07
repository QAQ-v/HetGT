# HetGT
An implementation for paper "Heterogeneous Graph Transformer for Graph-to-Sequence Learning" (accepted at ACL20).

## Requirements
```pip install -r requirements.txt```

## Preprocessing
Transform the input graphs into Levi graphs. The code is modified from Beck et al. (2018) and Guo et al. (2019)

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
```
@inproceedings{yao-etal-2020-heterogeneous,
    title = "Heterogeneous Graph Transformer for Graph-to-Sequence Learning",
    author = "Yao, Shaowei  and
      Wang, Tianming  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.640",
    doi = "10.18653/v1/2020.acl-main.640",
    pages = "7145--7154",
    abstract = "The graph-to-sequence (Graph2Seq) learning aims to transduce graph-structured representations to word sequences for text generation. Recent studies propose various models to encode graph structure. However, most previous works ignore the indirect relations between distance nodes, or treat indirect relations and direct relations in the same way. In this paper, we propose the Heterogeneous Graph Transformer to independently model the different relations in the individual subgraphs of the original graph, including direct relations, indirect relations and multiple possible relations between nodes. Experimental results show that our model strongly outperforms the state of the art on all four standard benchmarks of AMR-to-text generation and syntax-based neural machine translation.",
}
```
