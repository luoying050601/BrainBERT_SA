#horovodrun -np 8 \
# shellcheck disable=SC2006
cur_date="`date "+%Y-%m-%d-%H:%M:%S"`"

nohup python3 src/com/pre_train/run_pretrain_brain_bert.py \
--config config/pretrain-mlm-alice.json \
>> "mlm_alice_ae_$cur_date".out 2>&1 &