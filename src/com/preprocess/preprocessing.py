"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess NLVR annotations into LMDB
"""
import argparse
import json
import os
import lmdb
import parser
import msgpack
import numpy as np
import pandas as pd
from lz4.frame import compress
from pytorch_pretrained_bert import BertTokenizer
from src.com.util.data_format import bert_tokenize,open_lmdb


# participants = {'18'}
# # participants = {'18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37',
# #                 '39', '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53'}
PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../"))




def process_pereira(db, tokenizer, output_path, missing=None, type='test'):
    id2len = {}
    txt2img = {}  # not sure if useful
    sentence1_tsv_path = '../../../txt/pereira/text_807.tsv'
    sentence_df_path = ''
    txt2brain = json.load(open(f'{output_path}/txt2brain.json', 'r'))
    if type == 'test':
        type1_participants = ['P01']
    else:
        type1_participants = ['P01', 'M02', 'M04', 'M07', 'M15']

    for txt_id, brain_id in txt2brain.items():
        sentence_id = int(txt_id.split('-')[2])
        user = txt_id.split('-')[1]
        example = {}
        if user in type1_participants:
            sentence_df_path = sentence1_tsv_path
        sentence_df = pd.read_csv(sentence_df_path, sep='\t', header=None)
        sentence_df.columns = ['text', 'type']
        input_ids = tokenizer(sentence_df['text'][sentence_id])
        if 'label' in example:
            target = 1 if example['label'] == 'True' else 0
        else:
            target = None
        id2len[txt_id] = len(input_ids)
        example['input_ids'] = input_ids
        example['sentence'] = sentence_df['text'][sentence_id]
        example['target'] = target
        example['img_fname'] = brain_id
        db[txt_id] = example

    return id2len, txt2img


def process_pereira_word(db, tokenizer, output_path, missing=None, type='pereira_word'):
    id2len = {}
    txt2img = {}  # not sure if useful
    # participants_pereira = {'P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
    #                         'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17'}
    sentence_df_path = '../../../txt/pereira/text_180.tsv'
    txt2brain = json.load(open(f'{output_path}/txt2brain.json', 'r'))

    for txt_id, brain_id in txt2brain.items():
        sentence_id = int(txt_id.split('-')[2])
        example = {}
        sentence_df = pd.read_csv(sentence_df_path, sep='\t', header=None)
        sentence_df.columns = ['text', 'type']
        input_ids = tokenizer(sentence_df['text'][sentence_id])
        if 'label' in example:
            target = 1 if example['label'] == 'True' else 0
        else:
            target = None
        id2len[txt_id] = len(input_ids)
        example['input_ids'] = input_ids
        example['sentence'] = sentence_df['text'][sentence_id]
        example['target'] = target
        example['img_fname'] = brain_id
        db[txt_id] = example

    return id2len, txt2img


def process_alice(db, tokenizer, output_path, missing=None):
    ann = PROJ_DIR + "/text_tmp/alice_sentence.tsv"
    id2len = {}
    txt2img = {}  # not sure if useful
    sentence_df = pd.read_csv(ann, sep='\t', header=0)
    txt2brain = json.load(open(f'{output_path}/txt2brain.json', 'r'))
    for txt_id, brain_id in txt2brain.items():
        sentence_id = int(txt_id.split('-')[2])
        # sub_id = txt_id.split('-')[1]
        example = {}

        # _type = sentence_df['type'][sentence_id]
        # id_ = _type + '-' + sub_id + '-' + str(sentence_id)

        input_ids = tokenizer(sentence_df['sentences'][sentence_id])
        if 'label' in example:
            target = 1 if example['label'] == 'True' else 0
        else:
            target = None
        id2len[txt_id] = len(input_ids)
        example['input_ids'] = input_ids
        example['sentence'] = sentence_df['sentences'][sentence_id]
        example['target'] = target
        example['img_fname'] = brain_id
        db[txt_id] = example

    return id2len, txt2img


def generate_brain_mdb(dataset_name, type):
    participants_pereira = {'P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
                            'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17'}
    feature_Path = '/Storage/ying/resources/BrainBertTorch/dataset/' + dataset_name + '/features/'
    norm_Path = '/Storage/ying/resources/BrainBertTorch/dataset/' + dataset_name + '/norm/'
    brain_data_Path = PROJ_DIR+'/brain/' + dataset_name + '/pretrain_' + type + '.db/'
    if os.path.exists(brain_data_Path + 'data.mdb'):
        # 删除文件,path为文件路径
        os.remove(brain_data_Path + 'data.mdb')
    env = lmdb.open(brain_data_Path, map_size=1099511627776 * 32)
    txn = env.begin(write=True)
    fileList = os.listdir(feature_Path)
    write_cnt = 0
    count = 0
    for i in range(len(fileList)):
        # brain_frame_id = int(fileList[i].replace('.npz', '').split('_')[2])
        # alice
        # pre_train 0, 282
        # test 282, 322
        # val 322, 362
        file_spilt = fileList[i].replace('.npz', '').split('_')
        index = int(file_spilt[2])
        user = file_spilt[1]
        # exp_index = file_spilt[2]
        brain_data = {}
        # brain_data = np.load(feature_Path + fileList[i])
        if user in participants_pereira:  # 807 type
            if type == 'train':
                start = 0
                end = 144
            elif type == 'test':
                start = 144
                end = 162
            else:
                start = 162
                end = 180
        # if user in participants_pereira:  # 807 type
        #     if type == 'train':
        #         start = 0
        #         end = 645
        #     elif type == 'test':
        #         start = 645
        #         end = 726
        #     else:
        #         start = 726
        #         end = 807
        if index in range(start, end):
            filename = fileList[i]
            if fileList[i].find('.npz') != -1:
                print(filename)
                count = 1 + count
                write_cnt = write_cnt + 1
                feature_data = np.load(feature_Path + filename, allow_pickle=True)
                norm_data = np.load(norm_Path + filename, allow_pickle=True)
                feature_data = dict(enumerate(feature_data['features'].flatten(), 1))[1]
                norm_data = dict(enumerate(norm_data['features'].flatten(), 1))[1]

                brain_data['features'] = feature_data

                brain_data['norm_bb'] = norm_data
                brain_data['conf'] = None
                brain_data['soft_labels'] = None
                # a = []

                brain_data = compress(msgpack.dumps(brain_data, use_bin_type=True))
                txn.put(filename.encode(), brain_data)
                # print(i, filename, str(brain_data).encode().__sizeof__())
                # brain_data = {}
                if write_cnt % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    write_cnt = 0

    txn.commit()
    env.close()
    print(count)


def create_text_mdb_alice():
    key_word = 'test'
    #
    type_list = ['train', 'val', 'test']
    for _type in type_list:
        output_path = 'txt/' + key_word + '/pretrain_' + _type + '.db'
        args = parser.parse_args()
        # args.output = f'txt/alice/pretrain_val.db/'
        # filePath = args.output
        if os.path.exists(output_path + 'data.mdb'):
            # 删除文件,path为文件路径
            # chmod(nbb_path, 777)
            os.remove(output_path + 'data.mdb')
        args.toker = 'bert-large-uncased'

        args.missing_imgs = None
        #
        bertTokenizer = BertTokenizer.from_pretrained(
            args.toker, do_lower_case='uncased' in args.toker)
        tokenizer = bert_tokenize(bertTokenizer)
        db = open_lmdb(output_path, readonly=False)
        if args.missing_imgs is not None:
            missing_imgs = set(json.load(open(args.missing_imgs)))
        else:
            missing_imgs = None
        id2lens, txt2img = process_alice(db, tokenizer, output_path, missing_imgs)
        with open(f'{output_path}/id2len.json', 'w') as f:
            json.dump(id2lens, f)
        f.close()

def create_text_mdb_pereira():
    key_word = 'test'
    #
    type_list = ['train', 'val', 'test']
    for _type in type_list:
        output_path = PROJ_DIR+'/txt/'+key_word+'/pretrain_' + _type + '.db'
        args = parser.parse_args()
        # args.output = f'txt/pereira/pretrain_val.db/'
        # filePath = args.output
        if os.path.exists(output_path + 'data.mdb'):
            # 删除文件,path为文件路径
            # chmod(nbb_path, 777)
            os.remove(output_path + 'data.mdb')
        args.toker = 'bert-large-uncased'

        args.missing_imgs = None
        #
        bertTokenizer = BertTokenizer.from_pretrained(
            args.toker, do_lower_case='uncased' in args.toker)
        tokenizer = bert_tokenize(bertTokenizer)
        db = open_lmdb(output_path, readonly=False)
        if args.missing_imgs is not None:
            missing_imgs = set(json.load(open(args.missing_imgs)))
        else:
            missing_imgs = None
        id2lens, txt2img = process_pereira(db, tokenizer, output_path, missing_imgs, type=key_word)
        with open(f'{output_path}/id2len.json', 'w') as f:
            json.dump(id2lens, f)
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



def create_text_mdb_pereira_word():
    key_word = 'pereira_word'
    #
    type_list = ['train', 'val', 'test']
    for _type in type_list:
        output_path = PROJ_DIR+'/txt/'+key_word+'/pretrain_' + _type + '.db'
        args = parser.parse_args()
        if os.path.exists(output_path + 'data.mdb'):
            os.remove(output_path + 'data.mdb')
        args.toker = 'bert-large-uncased'
        args.missing_imgs = None
        bertTokenizer = BertTokenizer.from_pretrained(
            args.toker, do_lower_case='uncased' in args.toker)
        tokenizer = bert_tokenize(bertTokenizer)
        db = open_lmdb(output_path, readonly=False)
        if args.missing_imgs is not None:
            missing_imgs = set(json.load(open(args.missing_imgs)))
        else:
            missing_imgs = None
        id2lens, txt2img = process_pereira_word(db, tokenizer, output_path, missing_imgs, type=key_word)
        if os.path.exists(f'{output_path}/id2len.json'):
            # 删除文件,path为文件路径
            os.remove(f'{output_path}/id2len.json')
        with open(f'{output_path}/id2len.json', 'w') as f:
            json.dump(id2lens, f)
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

def generate_brain2text():
    type_list = ['pre_train', 'val', 'test']

    for _type in type_list:
        brain2txt = {}
        output_path = 'txt/alice/pretrain_' + _type + '.db'
        txt2brain = json.load(open(f'{output_path}/txt2brain.json', 'r'))
        for txt_id, brain_id in txt2brain.items():
            # for brain in brain_list:
            brain2txt[brain_id] = txt_id
        with open(f'{output_path}/brain2txt.json', 'w') as f:
            json.dump(brain2txt, f)
        f.close()

def create_text_mdb(type):
    if type == 'alice':
        create_text_mdb_alice()
    elif type == 'pereira':
        create_text_mdb_pereira()
    elif type == 'pereira_word':
        create_text_mdb_pereira_word()
    elif type == 'bookcorpus':
        create_text_mdb_pereira_word()
    elif type == 'wikicorpus':
        create_text_mdb_pereira_word()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # create_text_mdb_alice()
    # create_text_mdb_pereira()
    # parser.add_argument('--annotation', required=True,
    #                     help='annotation JSON')
    # parser.add_argument('--missing_imgs',
    #                     help='some training image features are corrupted')
    #                     help='some training image features are corrupted')
    # parser.add_argument('--output', required=True,
    #                     help='output dir of DB')
    # parser.add_argument('--toker', default='bert-base-cased',
    #                     help='which BERT tokenizer to used')

    # main()
    create_text_mdb_pereira_word()
    # 用这个生成的brain mdb
    generate_brain_mdb(dataset_name='pereira_word', type='train')
    generate_brain_mdb(dataset_name='pereira_word', type='test')
    generate_brain_mdb(dataset_name='pereira_word', type='val')
    # generate_brain_mdb('test')
    # generate_brain2text()
