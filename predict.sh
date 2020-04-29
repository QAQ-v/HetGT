set -e
MODEL="${1}"
DATA_DIR="${2:-data/amr2015}"
gpu="${3:-6}"
TGT="${4:-cs}"

CUDA_VISIBLE_DEVICES=$gpu

result=$(echo $DATA_DIR | grep "bpe")

if [[ $result != "" ]]; then
    echo "translate bpe data"

    python translate.py -model ${MODEL} -src ${DATA_DIR}/test.en.tok.bpe -grh ${DATA_DIR}/test.en.deps.bpe -output ${MODEL}.pred.bpe -replace_unk -share_vocab -beam_size 5 -gpu 0

    sed -r 's/(@@ )|(@@ ?$)//g' ${MODEL}.pred.bpe > ${MODEL}.pred.bpe.recover

    #python postprocess.py data/amr2015/test_map.pp.txt ${MODEL}.pred.bpe.recover ${MODEL}.final

    echo "-------------------"

else
    echo "translate non-bpe data"

    python translate.py -model ${MODEL} -src ${DATA_DIR}/test.amr -grh ${DATA_DIR}/test.grh -output ${MODEL}.pred -replace_unk -share_vocab -beam_size 5 -gpu 0 -log_file result # still need logfile, or change logger.info() to print()

    python postprocess.py ${DATA_DIR}/test_map.pp.txt ${MODEL}.pred ${MODEL}.final

    echo "-------------------"

fi

perl tools/multi-bleu.perl data/nmt19/en2${TGT}/test.${TGT}.tok  < ${MODEL}.pred.bpe.recover


echo "-------------------"
echo "test" ${MODEL} "done!"
echo "-------------------"