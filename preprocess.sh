DATA_DIR="${1:-data/amr2015}"
VOCAB_SIZE="${2:-5000}"

if [ ! -d ${DATA_DIR} ]; then
    mkdir ${DATA_DIR}
fi

#cat ${DATA_DIR}/train.amr.bpe data/amr2015.bpe.10/dev.amr.bpe > ${DATA_DIR}/train.amr.bpe.both
#cat ${DATA_DIR}/train.snt.bpe data/amr2015.bpe.10/dev.snt.bpe > ${DATA_DIR}/train.snt.bpe.both
#cat ${DATA_DIR}/train.grh.bpe data/amr2015.bpe.10/dev.grh.bpe > ${DATA_DIR}/train.grh.bpe.both

if [[ ${DATA_DIR} =~ bpe ]]; then
    echo "train bpe model"
    python preprocess.py   -train_src ${DATA_DIR}/train.en.tok.new.bpe \
                       -train_tgt ${DATA_DIR}/train.cs.tok.bpe \
                       -train_graph ${DATA_DIR}/train.en.deps.new.bpe \
                       -valid_src ${DATA_DIR}/val.en.tok.new.bpe  \
                       -valid_tgt ${DATA_DIR}/val.cs.tok.bpe \
                       -valid_graph ${DATA_DIR}/val.en.deps.new.bpe \
                       -edges_vocab ${DATA_DIR}/edge_vocab.json \
                       -save_data ${DATA_DIR}/amr2015 \
                       -src_vocab_size 30000  \
                       -tgt_vocab_size 30000 \
                       -shared_vocab_size 20000 \
                       -src_seq_length 10000 \
                       -tgt_seq_length 10000 \
                       -share_vocab
else
    echo "train non-bpe model"
    python preprocess.py   -train_src ${DATA_DIR}/train.amr \
                       -train_tgt ${DATA_DIR}/train.snt \
                       -train_graph ${DATA_DIR}/train.grh \
                       -valid_src ${DATA_DIR}/dev.amr  \
                       -valid_tgt ${DATA_DIR}/dev.snt \
                       -valid_graph ${DATA_DIR}/dev.grh \
                       -edges_vocab ${DATA_DIR}/edge_vocab.json \
                       -save_data ${DATA_DIR}/amr2015 \
                       -src_vocab_size 30000 \
                       -tgt_vocab_size 30000 \
                       -shared_vocab_size ${VOCAB_SIZE} \
                       -src_seq_length 10000 \
                       -tgt_seq_length 10000 \
                       -share_vocab
fi