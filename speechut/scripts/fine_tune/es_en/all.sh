# ####################################
# SpeechUT Base model #

#5.6 Baseline
# ####################################

w2v_path=//zhang_m/model/pretrain/checkpoint_204_400000.pt
DATA_DIR=/zhang_m/covost/es
lang=es
world_size=4
update_freq=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
CODE_ROOT=/home/SpeechUT
MODEL_DIR=/zhang_m/model/es_en/ambigst_new1
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

max_tokens=800000
python $CODE_ROOT/fairseq/fairseq_cli/train.py ${DATA_DIR} \
    --save-dir ${MODEL_DIR} \
    --user-dir /zhang_m/align/speechut \
    --task speech_to_text_new_acc_base \
    --config-yaml config_esen.yaml \
    --train-subset "train_st_homophone_seg_plus" \
    --valid-subset "dev_st" \
    --fp16 \
    --seed 1 \
    \
    --ddp-backend no_c10d \
    --distributed-world-size ${world_size} \
    --tensorboard-logdir ${MODEL_DIR} \
    --criterion multi_task_cross_entropy --report-accuracy \
    --label-smoothing 0.3 \
    --optimizer adam \
    --clip-norm 1.0 \
    --lr 3e-05 \
    --lr-scheduler polynomial_decay --warmup-updates 3000 \
    --max-update 22000 \
    --total-num-update 22000 \
    --update-freq ${update_freq} \
    --max-tokens ${max_tokens} \
    --max-sentences 16 \
    --max-tokens-valid ${max_tokens} \
    --grouped-shuffling \
    --max-source-positions ${max_tokens} \
    --skip-invalid-size-inputs-valid-test \
    --num-workers 16 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --arch "speechut_st_legacy" \
    --w2v-path ${w2v_path} \
    --layerdrop 0.1 \
    --activation-dropout 0.1 \
    --attention-dropout 0.1 \
    --feature-grad-mult 1.0 \
    --apply-mask --mask-prob 0.05 \
    --freeze-pretrain -1 \
    --log-format json \
    --log-interval 100 \
    --save-interval-updates 500 \
    --save-interval 1 \
    --keep-last-epochs 5 \
    --keep-interval-updates 10 \
    --keep-best-checkpoints 10 \
    
    \
    2>&1 | tee ${MODEL_DIR}/train_en${lang}.log


