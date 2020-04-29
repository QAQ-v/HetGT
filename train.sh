DATA_DIR="${1:-data/amr2015/amr2015}"
MODEL_DIR="${2:-model/amr2015/multiview_cat}"
GPU="${3:-0}"
LOG="${4}"
FUSION="${5-None}"

CUDA_VISIBLE_DEVICES=$GPU

echo   train.py -data ${DATA_DIR} -save_model ${MODEL_DIR}/ADAM \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 300000  -max_generator_batches 100 -dropout 0.3 \
        -batch_size 4096 -batch_type tokens -normalization tokens  \
        -optim adam -adam_beta2 0.998 -decay_method noam \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 5000 -save_checkpoint_steps 5000 \
        -world_size 1 -gpu_ranks 0 --report_every 100 \
        --share_decoder_embeddings --share_embeddings -aggregation $FUSION \
        -log_file $LOG -accum_count 2 -warmup_steps 8000 -learning_rate 2 -edges 5
        #-train_from model/amr2017/multiview_cat.bpe10.jump5.lr2.both/ADAM_step95000_lr0.00029.pt

python  train.py -data ${DATA_DIR} -save_model ${MODEL_DIR}/ADAM \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 300000  -max_generator_batches 100 -dropout 0.3 \
        -batch_size 4096 -batch_type tokens -normalization tokens  \
        -optim adam -adam_beta2 0.998 -decay_method noam \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 5000 -save_checkpoint_steps 10000 \
        -world_size 1 -gpu_ranks 0 --report_every 100 \
        --share_decoder_embeddings --share_embeddings -aggregation $FUSION \
        -log_file $LOG -accum_count 2 -warmup_steps 8000 -learning_rate 2 -edges 5
        #-train_from model/amr2017/multiview_cat.bpe10.jump5.lr2.both/ADAM_step95000_lr0.00029.pt
