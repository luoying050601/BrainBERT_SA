#horovodrun -np 8 \
# shellcheck disable=SC2006
cur_date="`date "+%Y-%m-%d-%H:%M:%S"`"
nohup python3 src/com/pre_train/run_pretrain_brain_bert.py \
--config config/pretrain-btm-base-8gpu.json \
>> "btm_$cur_date".out 2>&1 &