import os
# serve
from dataclasses import dataclass, field
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
import sys
sys.path.append(Proj_dir)
# local
# Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
from transformers import (
    HfArgumentParser,
    TrainingArguments
)
from datasets import load_dataset

import parser
from pytorch_pretrained_bert import BertTokenizer
from src.com.util.data_format import bert_tokenize,open_lmdb,preprocess,split_into_sentences
from src.com.util.logger import make_print_to_file
import json
import argparse
import time

def process_text(database,output_path,tokenizer):
    # 加载json
    text_json = json.load(open(f'{output_path}/corpus.json', 'r'))
    id2len = {}
    example = {}
    for k,v in text_json.items():
        sentence = preprocess(v)
        # 计算len
        input_ids = tokenizer(sentence)
        id2len[k] = len(input_ids)
        example['input_ids'] = input_ids
        example['sentence'] = sentence
        example['target'] = None
        example['img_fname'] = None
        database[k] = example
    # 生成token 保存到lmdb下
    return id2len


def create_corpus_json(key_word,output_path, type,is_test):
    ratio = 1
    text_list = []
    if type == 'train':
        ratio = 0.8
    elif type == 'val':
        ratio = 0.9
    if key_word == 'bookcorpus':
        corpus = load_dataset(key_word)
        text_list = corpus['train']['text']

    elif key_word == 'wikicorpus':
        corpus = load_dataset(key_word, 'raw_en')
        for sentences in corpus['train']['text']:
            if len(sentences) < 512:
                text_list.append(sentences)
            # text_list.extend(split_into_sentences(preprocess(passage)))
#     ...
    if is_test:
        text_list = text_list[:50000]
    start = 0
    if type == 'val':
        start = int(0.8 * len(text_list))
    elif type == 'test':
        start = int(0.9 * len(text_list))
    end = int(ratio * len(text_list))
    corpus_json = {}
    # print(key_word,type,start,end)
    for i in range(start,end):
        # test-18-67-0
        # print(key_word,type,i)
        text_id = type+'-'+ str(i)
        sentence = text_list[i]
        # print(sentence)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        corpus_json[text_id] = preprocess(sentence)
    with open(f'{output_path}/corpus.json', 'w') as f:
            json.dump(corpus_json, f)

def create_text_mdb(key_word,_type,is_test=False):
        if is_test:
            output_path = '/home/sakura/resources/BrainBertTorch/txt/' + key_word + '_test/pretrain_' + _type + '.db'
        else:
            output_path = '/home/sakura/resources/BrainBertTorch/txt/' + key_word + '/pretrain_' + _type + '.db'
        if not os.path.exists(f'{output_path}/corpus.json'):
            create_corpus_json(key_word=key_word,output_path=output_path,type=_type, is_test=is_test)
        else:
            args = parser.parse_args()
            if os.path.exists(output_path + 'data.mdb'):
                os.remove(output_path + 'data.mdb')
            args.toker = 'bert-large-uncased'
            bertTokenizer = BertTokenizer.from_pretrained(
                args.toker, do_lower_case='uncased' in args.toker)
            tokenizer = bert_tokenize(bertTokenizer)
            db = open_lmdb(output_path, readonly=False)

            id2len = process_text(database=db,output_path=output_path,tokenizer=tokenizer)
            if os.path.exists(f'{output_path}/id2len.json'):
                # 删除文件,path为文件路径
                os.remove(f'{output_path}/id2len.json')
            with open(f'{output_path}/id2len.json', 'w') as f:
                json.dump(id2len, f)
            f.close()
            meta = vars(args)
            meta['tokenizer'] = args.toker
            meta['UNK'] = bertTokenizer.convert_tokens_to_ids(['[UNK]'])[0]
            meta['CLS'] = bertTokenizer.convert_tokens_to_ids(['[CLS]'])[0]
            meta['SEP'] = bertTokenizer.convert_tokens_to_ids(['[SEP]'])[0]
            meta['MASK'] = bertTokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            meta['v_range'] = (bertTokenizer.convert_tokens_to_ids('!')[0],
                               len(bertTokenizer.vocab))
            with open(f'{output_path}/meta.json', 'w') as f:
                json.dump(vars(args), f, indent=4)


    # process_text(key_word=key_word)
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    key_word: str = field(
        metadata={"help": ""}
    )
    type: str = field(
        default="train",
        metadata={"help": ""}
    )

    is_test: bool = field(
        default=True,
        metadata={"help": ""}
    )


if __name__ == '__main__':
    start = time.perf_counter()
    # make_print_to_file(path='.')
    parser = HfArgumentParser(ModelArguments)
    model_args= parser.parse_args_into_dataclasses()
    print(model_args)
    # type_list = ['train', 'val', 'test']

    # create_text_mdb('bookcorpus','val')
    # create_text_mdb('bookcorpus','test')
    create_text_mdb(model_args[0].key_word,model_args[0].type, is_test=model_args[0].is_test)
    # create_text_mdb('wikicorpus','val')
    # create_text_mdb('wikicorpus','train')
    # create_text_mdb('wikicorpus','test')

    end = time.perf_counter()
    time_cost = str((end - start) / 60)
    print("time-cost:", time_cost)

