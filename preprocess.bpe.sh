set -e
OPERATIONS="${1:-10000}"
TGT="${2:-de}"
DATA_DIR=data/nmt19/en2$TGT
FINAL_DIR=${DATA_DIR}.bpe.${OPERATIONS}.both.new
echo $DATA_DIR
if [ ! -d ${FINAL_DIR} ]; then
    mkdir ${FINAL_DIR}
fi

cat ${DATA_DIR}/train.en.tok.new ${DATA_DIR}/val.en.tok.new > ${FINAL_DIR}/train.en.tok.both
cat ${DATA_DIR}/train.${TGT}.tok ${DATA_DIR}/val.${TGT}.tok > ${FINAL_DIR}/train.${TGT}.tok.both
#cat ${DATA_DIR}/train.en.deps.new ${DATA_DIR}/val.en.deps.new > ${FINAL_DIR}/train.en.deps.both

cat ${FINAL_DIR}/train.en.tok.both ${FINAL_DIR}/train.${TGT}.tok.both > ${FINAL_DIR}/train.for.codes

subword-nmt learn-bpe -s $OPERATIONS < ${FINAL_DIR}/train.for.codes > ${FINAL_DIR}/train.codes
subword-nmt apply-bpe -c ${FINAL_DIR}/train.codes < ${DATA_DIR}/train.en.tok.new > ${FINAL_DIR}/train.en.tok.new.bpe
subword-nmt apply-bpe -c ${FINAL_DIR}/train.codes < ${DATA_DIR}/train.${TGT}.tok > ${FINAL_DIR}/train.${TGT}.tok.bpe
subword-nmt apply-bpe -c ${FINAL_DIR}/train.codes < ${DATA_DIR}/val.en.tok.new > ${FINAL_DIR}/val.en.tok.new.bpe
subword-nmt apply-bpe -c ${FINAL_DIR}/train.codes < ${DATA_DIR}/val.${TGT}.tok > ${FINAL_DIR}/val.${TGT}.tok.bpe
subword-nmt apply-bpe -c ${FINAL_DIR}/train.codes < ${DATA_DIR}/test.en.tok.new > ${FINAL_DIR}/test.en.tok.bpe
#subword-nmt apply-bpe -c ${FINAL_DIR}/train.codes < ${DATA_DIR}/test.${TGT}.tok > ${FINAL_DIR}/test.${TGT}.tok.bpe

python bpe_dege.py ${DATA_DIR}/train.en.deps.new ${DATA_DIR}/train.en.tok.new ${FINAL_DIR}/train.en.tok.new.bpe ${FINAL_DIR}/train.en.deps.new.bpe
python bpe_dege.py ${DATA_DIR}/val.en.deps.new ${DATA_DIR}/val.en.tok.new ${FINAL_DIR}/val.en.tok.new.bpe ${FINAL_DIR}/val.en.deps.new.bpe
python bpe_dege.py ${DATA_DIR}/test.en.deps.new ${DATA_DIR}/test.en.tok.new ${FINAL_DIR}/test.en.tok.bpe ${FINAL_DIR}/test.en.deps.bpe
cp ${DATA_DIR}/edge_vocab.json ${FINAL_DIR}/
