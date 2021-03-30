#horovodrun -np 8 \
# shellcheck disable=SC2006
cur_date="`date "+%Y-%m-%d-%H:%M:%S"`"
    # type_list = ['train', 'val', 'test']

nohup python3 src/com/text/load_huggingface_dataset.py \
--type 'train' \
--key_word 'bookcorpus' \
--is_test true\
>> "bookcorpus_train_$cur_date".out 2>&1 &
#
nohup python3 src/com/text/load_huggingface_dataset.py \
--type 'val' \
--key_word 'bookcorpus' \
--is_test true\
>> "bookcorpus_val_$cur_date".out 2>&1 &

nohup python3 src/com/text/load_huggingface_dataset.py \
--type 'test' \
--key_word 'bookcorpus' \
--is_test true\
>> "bookcorpus_test_$cur_date".out 2>&1 &

#nohup python3 src/com/text/load_huggingface_dataset.py \
#--type 'train' \
#--key_word 'wikicorpus' \
#>> "wikicorpus_train_$cur_date".out 2>&1 &
#
#nohup python3 src/com/text/load_huggingface_dataset.py \
#--type 'val' \
#--key_word 'wikicorpus' \
#>> "wikicorpus_val_$cur_date".out 2>&1 &
#
#nohup python3 src/com/text/load_huggingface_dataset.py \
#--type 'test' \
#--key_word 'wikicorpus' \
#>> "wikicorpus_test_$cur_date".out 2>&1 &