import os
import json
import pandas as pd
from src.com.util.data_format import get_word, count_words, getMaxCommonSubstr
import torch
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
orginal_txt_data = PROJ_DIR + "/text_tmp/alice_in_wonderland.txt"
orginal_alice_tsv = PROJ_DIR + "/text_tmp/annotations.tsv"
orginal_brain_tsv = PROJ_DIR + "/text_tmp/annotations_brain.tsv"
sentence_tsv_path = PROJ_DIR + "/text_tmp/alice_sentence.tsv"
s2b_tsv_path = PROJ_DIR + "/text_tmp/alice_sent_to_brain.tsv"
output_alice_json = PROJ_DIR + "/text_tmp/alice_text_brain.json"
model_file = PROJ_DIR + '/model/autoencoder_mri_kf.h5'
sent_to_brain_json = PROJ_DIR + "/text_tmp/alice_sent_to_brain.json"
dataset_name = "alice"
participants_alice = {'18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37',
                      '39', '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53'}
participants_pereira = {'P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
                        'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17'}


# create sentence tsv file
def create_sentence_file():
    # in_path = orginal_txt_data
    # out_path = output_sentence_path
    from nltk import tokenize
    sentence_df = pd.DataFrame(columns=["id", "type", "onset", "offset", "sentences"])

    with open(orginal_txt_data, "r") as f:
        for line in f:
            l = tokenize.sent_tokenize(line)
            train_group = (len(l) / 10) * 8
            test_group = len(l) / 10 + train_group
            alice_df = pd.read_csv(orginal_alice_tsv, sep='\t', header=0)
            mark_index = 0

            for i in range(len(l)):
                s = l[i].lower().replace('ma\'am', 'mama'). \
                    replace('n\'t', ' nt').replace("\'", " "). \
                    replace(")", " ").replace("(", "")
                sentence_df.loc[i, 'sentences'] = s
                if i < train_group:
                    _type = "pre_train"
                elif i < test_group:
                    _type = "test"
                else:
                    _type = "val"
                sentence_df.loc[i, 'type'] = _type
                first_word = get_word(s.replace("?", " ").replace("!", " "), 0)
                last_word = get_word(s.replace("?", " ").replace("!", " "), -1)

                word_count = count_words(s)
                for al in range(mark_index, len(alice_df)):
                    # if i >= 80:
                    # print(350)
                    if al < mark_index - 2:
                        alice_df.loc[al, 'Sentence Id'] = i
                        continue
                    else:
                        alice_df.loc[al, 'Sentence Id'] = i

                    if pd.isna(sentence_df['onset'][i]):
                        if first_word == alice_df['Words'][al].lower():
                            if not pd.isna(alice_df['Word Onset'][al]):
                                sentence_df.loc[i, 'onset'] = int((alice_df['Word Onset'][al] - 20) / 2)
                                alice_df.loc[al, 'Sentence Id'] = i
                                mark_index = al + word_count
                    if not pd.isna(sentence_df['onset'][i]) and pd.isna(sentence_df['offset'][i]) and \
                            al in range(mark_index - 10, mark_index + 10):

                        if alice_df['Words'][al].lower() == last_word:
                            if not pd.isna(alice_df['Word Onset'][al]):
                                sentence_df.loc[i, 'offset'] = int((alice_df['Word Onset'][al] - 20) / 2)
                                alice_df.loc[al, 'Sentence Id'] = i
                                mark_index = al + 1
                    if not pd.isna(sentence_df['onset'][i]) and not pd.isna(sentence_df['offset'][i]):
                        break

    sentence_df.to_csv(sentence_tsv_path, sep='\t')
    alice_df.to_csv(orginal_alice_tsv, sep='\t', index=False)
    f.close()


def get_word_from_sentence(i1, i2):
    alice_df = pd.read_csv(orginal_alice_tsv, sep='\t', header=0)
    s_id = alice_df["Sentence Id"][i2]
    if s_id == i1:
        return (alice_df["Words"][i2]).lower().replace("'", "")
    return "None"


def get_sentence_embedding(sentence, option):
    # if YOU NEED [cls] and [seq] PRESENTATION：True
    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=False)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)

    all_layers_output = outputs[2]
    # list of torch.FloatTensor (one for the output of each layer +
    # the output of the embeddings) of shape (batch_size, sequence_length, hidden_size):
    # Hidden-states of the model at the output of each layer plus the initial embedding outputs.

    if option == "last_layer":
        sent_embeddings = all_layers_output[-1]  # last layer
    elif option == "second_to_last_layer":
        sent_embeddings = all_layers_output[-2]  # second to last layer
    else:
        sent_embeddings = all_layers_output[-1]  # last layer

    sent_embeddings = torch.squeeze(sent_embeddings, dim=0)
    # print(sent_embeddings.shape)

    # Calculate the average of all token vectors.
    sentence_embedding_avg = torch.mean(sent_embeddings, dim=0)
    # print(sentence_embedding_avg.shape)

    return sent_embeddings.detach(), sentence_embedding_avg.detach()


def create_txt_npz():
    import numpy as np
    import json

    # --- load the model ----

    npz_path = PROJ_DIR + '/txt/alice/pretrain_train.db/features'
    id_path = PROJ_DIR + '/txt/alice/pretrain_train.db/id2len.json'
    with open(id_path, 'r', encoding='utf-8') as f:
        content = f.read()
        a = json.loads(content)
        count = 0
        for i, k in enumerate(a):
            count = count + 1
            txt_id_list = k.split('-')
            word = get_word_from_sentence(int(txt_id_list[2]), int(txt_id_list[3]))
            # print(int(txt_id_list[2]), word)
            bert_layer_option = "last_layer"
            word_embeddings, sent_embedding_avg = get_sentence_embedding(word, bert_layer_option)
            npz_filename = k + '.npz'
            # print(npz_filename)
            #     #
            np.savez(npz_path + '/' + npz_filename, word_embeddings.numpy())
        # print(count)
    f.close()
    # # for sub_id in participants:
    # for txt_id in range(len(txt_id_df)):
    #     # for i in range(len(label)):
    #     txt_id_list = txt_id_df[txt_id].split('-')
    #     print(txt_id_list)
    #     # test-24-67-1664


#

def create_brain_npz():
    import tensorflow.keras as keras
    from src.com.util.data_format import noramlization, generate_array_from_file
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.models import load_model
    IMAGE_HEIGHT = 72
    IMAGE_WIDTH = 96
    IMAGE_DEPTH = 64
    model = load_model(model_file)
    # participants = ['42']
    pretrained_model = keras.Model(inputs=model.input,
                                   outputs=model.get_layer("latent").output)  # 你创建新的模型
    norm_path = PROJ_DIR + '/dataset/alice/norm/'
    feature_path = PROJ_DIR + '/dataset/alice/features/'
    feature_data = {}
    norm_data = {}
    for sub_id in participants_alice:
        tf_path = PROJ_DIR + '/dataset/alice/tf_data/test_data_vol_' + sub_id + '.tfrecords'
        test, label = generate_array_from_file(tf_path, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)
        # print(sub_id, len(label))
        # print(test)
        train_X = np.array(test).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])
        train_X, _, _ = noramlization(train_X)
        encoded_imgs = pretrained_model.predict(train_X)
        encoded_imgs, _, _ = noramlization(encoded_imgs)
        z = tf.reshape(encoded_imgs, [-1, 1024])
        norm_shape = test[0].shape
        feature_shape = z[0].shape[0]
        for i in range(len(label)):
            npz_filename = dataset_name + '_' + sub_id + '_' + str(i) + '.npz'
            norm_data['norm'] = {'data': test[i].tolist(),
                                 'shape': list(norm_shape),
                                 'type': None,
                                 'kind': None}
            feature_data['features'] = {'data': z[i].numpy().tolist(),
                                        'shape': [feature_shape],
                                        'type': None,
                                        'kind': None}
            np.savez(norm_path + npz_filename, norm_data=norm_data['norm'])
            np.savez(feature_path + npz_filename, features=feature_data['features'])


def create_text_json():
    import json
    sentence_df = pd.read_csv(sentence_tsv_path, sep='\t', header=0)
    orginal_df = pd.read_csv(orginal_alice_tsv, sep='\t', header=0)
    s2b_df = pd.read_csv(s2b_tsv_path, sep='\t', header=0)
    count = 0
    # print(pd.isna(orginal_df['Sentence Id']).sum())
    data = []
    t2b = {}
    name_list = []
    for s in range(len(sentence_df)):
        for b in range(len(s2b_df)):
            if s2b_df['sentence_id'][b] == s:
                name_list.append(str(s2b_df['brain_frame_id'][b]) + '.features')
        t2b[s] = name_list
        name_list = []

    with open(output_alice_json, 'w', encoding='utf-8') as fp:
        for sub_id in participants_alice:
            for al in range(len(orginal_df)):
                s = int(orginal_df['Sentence Id'][al])

                result = {
                    'id': count,
                    'name': sentence_df['type'][s] + '-' + sub_id + '-' + str(s) + '-' + str(al),
                    'sub_id': sub_id,
                    'sentence_id': s,
                    'word_id': al,
                    'word': orginal_df['Words'][al].lower(),
                    'len': count_words(sentence_df['sentences'][s]),
                    'text_to_brain': [dataset_name + '_' + sub_id + '_' + i for i in t2b[s]]
                    # dataset_name + '_' + sub_id + '_'+t2b[s]
                }
                count = count + 1
                data.append(result)
        json.dump(data, fp, ensure_ascii=False)


def sent_to_brain2():
    s2b_df = pd.DataFrame(columns=["sentence_id", "brain_frame_id"])

    # brain_df = pd.read_csv(orginal_brain_tsv, sep='\t', header=0)
    sentence_df = pd.read_csv(sentence_tsv_path, sep='\t', header=0)
    # data = []
    # begin = 0
    # end = len(brain_df)
    for sentence_id in range(len(sentence_df)):
        begin = int(sentence_df['onset'][sentence_id])
        end = int(sentence_df['offset'][sentence_id])
        for i in range(begin, end + 1):
            s2b_df.loc[i, 'sentence_id'] = sentence_id
            s2b_df.loc[i, 'brain_frame_id'] = i
    s2b_df.to_csv(s2b_tsv_path, sep='\t', index=False)

    # sent_onset = (sentence_df['onset'][sentence_id] + 10) * 2
    # sent_offset = (sentence_df['offset'][sentence_id] + 10) * 2


def sent_to_brain():
    # import json
    s2b_df = pd.DataFrame(columns=["sentence_id", "brain_frame_id"])

    brain_df = pd.read_csv(orginal_brain_tsv, sep='\t', header=0)
    sentence_df = pd.read_csv(sentence_tsv_path, sep='\t', header=0)
    data = []
    begin = 0
    end = len(brain_df)
    # with open(sent_to_brain_json, 'w', encoding='utf-8') as fp:
    for sentence_id in range(len(sentence_df) - 1, 0, -1):
        for b in range(end - 1, begin, -1):
            if sentence_id == 65:  # 16 32
                if b == 279:
                    print(1)

            sub_str = brain_df['Sentences'][b].lower().replace(",", "")
            sent_str = sentence_df['sentences'][sentence_id].replace("?", "").replace("!", "").replace(",", "")
            lenMatch, strMatch = getMaxCommonSubstr(sub_str, sent_str)
            if lenMatch >= 2 or sent_str.replace(" ", "") == 'well' or sent_str == 'thump':
                sub_onset = brain_df['Word Onset'][b]
                sent_onset = (sentence_df['onset'][sentence_id] + 10) * 2
                sent_offset = (sentence_df['offset'][sentence_id] + 10) * 2
                if sent_onset - 1 <= sub_onset <= sent_offset + 3:
                    map = [sentence_id, b]
                    if len(data) == 0:
                        data.append(map)
                    else:
                        # l = len(data)
                        flag = True
                        for k in range(len(data)):
                            if b == data[k][1]:
                                flag = False
                        if flag:
                            data.append(map)

    for i in range(len(data)):
        s2b_df.loc[i, 'sentence_id'] = data[i][0]
        s2b_df.loc[i, 'brain_frame_id'] = data[i][1]
    s2b_df.to_csv(s2b_tsv_path, sep='\t')
    # json.dump(data, fp, ensure_ascii=False)

def create_txt2brain_alice():
    import json
    s2b_df = pd.read_csv(s2b_tsv_path, sep='\t', header=0)
    sentence_df = pd.read_csv(sentence_tsv_path, sep='\t', header=0)
    orginal_df = pd.read_csv(orginal_alice_tsv, sep='\t', header=0)
    result_test = {}
    result_val = {}
    result_tra = {}
    data_test = {}
    data_tra = {}
    data_val = {}

    result = {}
    data = {}
    # if sentence_df['type'][sentence_id] == "dev":
    txt2brain = PROJ_DIR + "/txt/alice/txt2brain.json"

    # txt2brain_test = PROJ_DIR + "/txt/alice/pretrain_test.db/txt2brain2.json"
    # elif
    # txt2brain_train = PROJ_DIR + "/txt/alice/pretrain_train.db/txt2brain2.json"
    # else:
    # txt2brain_val = PROJ_DIR + "/txt/alice/pretrain_val.db/txt2brain2.json"
    brain_npz = {}
    for sub_id in participants_alice:
        for word_id in range(len(orginal_df)):
            sentence_id = int(orginal_df['Sentence Id'][word_id])
            _type = sentence_df['type'][sentence_id]
            # txt_id = _type + '-' + sub_id + '-' + str(sentence_id) + '-' + str(
            #     word_id)
            _list = []
            for i in range(len(s2b_df)):
                if s2b_df['sentence_id'][i] == sentence_id:
                    _list.append(dataset_name + '_' + sub_id + '_' + str(s2b_df['brain_frame_id'][i]) + '.npz')
            brain_npz[str(sub_id) + '_' + str(sentence_id)] = _list
            _list = []
    # f1 = open(txt2brain_test, 'w', encoding='utf-8')
    # f2 = open(txt2brain_train, 'w', encoding='utf-8')
    # f3 = open(txt2brain_val, 'w', encoding='utf-8')
    f4 = open(txt2brain, 'w', encoding='utf-8')
    for sub_id in participants_alice:
        for word_id in range(len(orginal_df)):
            sentence_id = int(orginal_df['Sentence Id'][word_id])
            _type = sentence_df['type'][sentence_id]
            txt_id = _type + '-' + sub_id + '-' + str(sentence_id)
            result[txt_id] = brain_npz[str(sub_id) + '_' + str(sentence_id)]
            data.update(result_test)
            if _type == "test":
                result_test[txt_id] = brain_npz[str(sub_id) + '_' + str(sentence_id)]
                data_test.update(result_test)
            if _type == "pre_train":
                result_tra[txt_id] = brain_npz[str(sub_id) + '_' + str(sentence_id)]
                data_tra.update(result_tra)
            if _type == "val":
                result_val[txt_id] = brain_npz[str(sub_id) + '_' + str(sentence_id)]
                data_val.update(result_val)

    # json.dump(data_test, f1, ensure_ascii=False)
    # json.dump(data_tra, f2, ensure_ascii=False)
    # json.dump(data_val, f3, ensure_ascii=False)
    json.dump(data_val, f4, ensure_ascii=False)
    # f1.close()
    # f2.close()
    # f3.close()
    f4.close()


def create_txt2brain_pereira():
    txt2brain_path = PROJ_DIR + "/txt/pereira/txt2brain.json"
    txt2brain_test = PROJ_DIR + "/txt/pereira/pretrain_test.db/txt2brain.json"
    txt2brain_train = PROJ_DIR + "/txt/pereira/pretrain_train.db/txt2brain.json"
    txt2brain_val = PROJ_DIR + "/txt/pereira/pretrain_val.db/txt2brain.json"
    result_test = {}
    result_val = {}
    result_tra = {}
    data_test = {}
    data_tra = {}
    data_val = {}
    txt2brain = json.loads(open(txt2brain_path, 'r', encoding='utf-8').read())
    f1 = open(txt2brain_test, 'w', encoding='utf-8')
    f2 = open(txt2brain_train, 'w', encoding='utf-8')
    f3 = open(txt2brain_val, 'w', encoding='utf-8')
    for text_id in txt2brain.keys():
        _type = text_id.split('-')[0]
        if _type == "test":
            result_test[text_id] = txt2brain[text_id]
            data_test.update(result_test)
        if _type == "pre_train":
            result_tra[text_id] = txt2brain[text_id]
            data_tra.update(result_tra)
        if _type == "val":
            result_val[text_id] = txt2brain[text_id]
            data_val.update(result_val)

    json.dump(data_test, f1, ensure_ascii=False)
    json.dump(data_tra, f2, ensure_ascii=False)
    json.dump(data_val, f3, ensure_ascii=False)
    # json.dump(data_val, f4, ensure_ascii=False)
    f1.close()
    f2.close()
    f3.close()


def create_brain2txt_pereira():
    sentence1_tsv_path = '../../../txt/pereira/text_807.tsv'
    sentence2_tsv_path = '../../../txt/pereira/text_564.tsv'
    sentence3_tsv_path = '../../../txt/pereira/text_423.tsv'
    sentence4_tsv_path = '../../../txt/pereira/text_180.tsv'
    # join exp1, 2and 3
    type1_participants = ['P01', 'M02', 'M04', 'M07', 'M15']
    # join exp1 and 3
    type2_participants = ['M08', 'M09', 'M14']
    # join exp1 and 3
    type3_participants = ['M03']
    # only join exp1
    type4_participants = ['M01', 'M05', 'M06', 'M10', 'M13', 'M16', 'M17']
    sentence_df_path = ''
    # for sub_id in participants_pereira:
    for path in ['pre_train', 'test', 'val']:
        brain2txt = PROJ_DIR + "/txt/pretrain_" + path + ".db/brain2txt.json"
        f = open(brain2txt, 'w', encoding='utf-8')
        result = {}
        data = {}
        txt2brain = json.loads(
            open(PROJ_DIR + '/txt/pretrain_' + path + '.db/txt2brain.json', 'r', encoding='utf-8').read())
        for brain_id in txt2brain.values():
            user = brain_id.replace('.npz', '').split('_')[1]
            # line_id = int(brain_id.replace('.npz', '').split('_')[2])
            if user in type1_participants:
                sentence_df_path = sentence1_tsv_path
            elif user in type2_participants:
                sentence_df_path = sentence2_tsv_path
            elif user in type3_participants:
                sentence_df_path = sentence3_tsv_path
            elif user in type4_participants:
                sentence_df_path = sentence4_tsv_path
            sentence_df = pd.read_csv(sentence_df_path, sep='\t', header=None)
            sentence_df.columns = ['text', 'type']
            # text_count = count_words(sentence_df['text'][line_id])
            # if text_count == 1:
            # 简单粗暴，brain 都对应句子第一个单词（反正embedding都是按照句子为单位）
            result[brain_id] = list(txt2brain.keys())[list(txt2brain.values()).index(brain_id)]
            #  text_list
            data.update(result)

        json.dump(data, f, ensure_ascii=False)
    f.close()


def create_brain2text(type):

    if type == 'test':
        sentence1_tsv_path = '../../../txt/pereira/text_807.tsv'
        type1_participants = ['P01']
        sentence_df_path = ''
        for path in ['train', 'test', 'val']:
            id2len_path = PROJ_DIR + "/txt/"+type+"/pretrain_" + path + ".db/brain2txt.json"
            f = open(id2len_path, 'w', encoding='utf-8')
            result = {}
            data = {}
            txt2brain = json.loads(
                open(PROJ_DIR + '/txt/'+type+'/pretrain_' + path + '.db/txt2brain.json', 'r', encoding='utf-8').read())
            for brain_id in txt2brain.values():
                user = brain_id.replace('.npz', '').split('_')[1]
                if user in type1_participants:
                    sentence_df_path = sentence1_tsv_path

                sentence_df = pd.read_csv(sentence_df_path, sep='\t', header=None)
                sentence_df.columns = ['text', 'type']

                result[brain_id] = list(txt2brain.keys())[list(txt2brain.values()).index(brain_id)]
                #  text_list
                data.update(result)


                # for i in range(len(sentence_df)):
            json.dump(data, f, ensure_ascii=False)
        f.close()
    elif type =='pereira_word':
        sentence4_tsv_path = '../../../txt/pereira/text_180.tsv'
        participants_pereira = {'P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
                                'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17'}
        sentence_df_path = ''
        for path in ['train', 'test', 'val']:
            brain2txt = PROJ_DIR + "/txt/" + type + "/pretrain_" + path + ".db/brain2txt.json"
            if os.path.exists(brain2txt):
                # 删除文件,path为文件路径
                os.remove(brain2txt)
            f = open(brain2txt, 'w', encoding='utf-8')
            result = {}
            data = {}
            txt2brain = json.loads(
                open(PROJ_DIR + '/txt/' + type + '/pretrain_' + path + '.db/txt2brain.json', 'r',
                     encoding='utf-8').read())
            for brain_id in txt2brain.values():
                user = brain_id.replace('.npz', '').split('_')[1]
                if user in participants_pereira:
                    sentence_df_path = sentence4_tsv_path
                sentence_df = pd.read_csv(sentence_df_path, sep='\t', header=None)
                sentence_df.columns = ['text', 'type']
                result[brain_id] = list(txt2brain.keys())[list(txt2brain.values()).index(brain_id)]
                #  text_list
                data.update(result)
            json.dump(data, f, ensure_ascii=False)
        f.close()



def create_text2brain(type):
    txt2brain_path = PROJ_DIR + "/txt/alice/txt2brain.json"
    txt2brain_test = PROJ_DIR + "/txt/" + type + "/pretrain_test.db/txt2brain.json"
    txt2brain_train = PROJ_DIR + "/txt/" + type + "/pretrain_train.db/txt2brain.json"
    txt2brain_val = PROJ_DIR + "/txt/" + type + "/pretrain_val.db/txt2brain.json"
    if os.path.exists(txt2brain_test):
        # 删除文件,path为文件路径
        os.remove(txt2brain_test)
    if os.path.exists(txt2brain_train):
            # 删除文件,path为文件路径
            os.remove(txt2brain_train)
    if os.path.exists(txt2brain_val):
        # 删除文件,path为文件路径
        os.remove(txt2brain_val)
    f1 = open(txt2brain_test, 'w', encoding='utf-8')
    f2 = open(txt2brain_train, 'w', encoding='utf-8')
    f3 = open(txt2brain_val, 'w', encoding='utf-8')
    txt2brain = json.loads(open(txt2brain_path, 'r', encoding='utf-8').read())
    result_test = {}
    result_val = {}
    result_tra = {}
    data_test = {}
    data_tra = {}
    data_val = {}
    if type == 'test':
        for text_id in txt2brain.keys():
            # val-P01-803-5
            _type = text_id.split('-')[0]
            user = text_id.split('-')[1]
            if user == 'P01':
                if _type == "test":
                    result_test[text_id] = txt2brain[text_id]
                    data_test.update(result_test)
                if _type == "train":
                    result_tra[text_id] = txt2brain[text_id]
                    data_tra.update(result_tra)
                if _type == "val":
                    result_val[text_id] = txt2brain[text_id]
                    data_val.update(result_val)
    elif type == 'pereira_word':
        # sentence4_tsv_path = 'txt/pereira/text_180.tsv'
        participants_pereira = {'P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
                                'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17'}
        for text_id in txt2brain.keys():
            # val-M01-803-5
            _type = text_id.split('-')[0]
            user = text_id.split('-')[1]
            word_id = int(text_id.split('-')[2])
            if word_id < 180:
                if user in participants_pereira:
                    if word_id < 144:
                        result_tra[text_id] = txt2brain[text_id]
                        data_tra.update(result_tra)
                    if word_id in range(144,162):
                        result_test[text_id] = txt2brain[text_id]
                        data_test.update(result_test)
                    if word_id in range(162,180):
                        result_val[text_id] = txt2brain[text_id]
                        data_val.update(result_val)
    json.dump(data_test, f1, ensure_ascii=False)
    json.dump(data_tra, f2, ensure_ascii=False)
    json.dump(data_val, f3, ensure_ascii=False)
    # json.dump(data_val, f4, ensure_ascii=False)
    f1.close()
    f2.close()
    f3.close()


if __name__ == "__main__":
    # create_text2brain('test')
    # create_brain2text('test')
    create_text2brain('alice_ae')
    create_brain2text('alice_ae')
    # create_text2brain('test')
    # create_brain2txt_pereira()
    # create_txt2brain_pereira()
#     import os
#
#     # 获取目标文件夹的路径
#     filedir1 = PROJ_DIR + "/txt/alice/pretrain_train.db/brain2txt.json"
#     filedir2 = PROJ_DIR + "/txt/alice/pretrain_val.db/brain2txt.json"
#     filedir3 = PROJ_DIR + "/txt/alice/pretrain_test.db/brain2txt.json"
#     # 获取当前文件夹中的文件名称列表
#     filenames = [filedir1, filedir2, filedir3]
#     # 打开当前目录下的result.json文件，如果没有则创建
#     f = open(PROJ_DIR + '/brain/alice/brain2txt.json', 'w')
#     # 先遍历文件名`在这里插入代码片`
#     for filename in filenames:
#         # filepath = filedir + '/' + filename
#         # 遍历单个文件，读取行数
#         for line in open(filename, "r"):
#             f.writelines(line)
#             f.write('\n')
#     # 关闭文件
#     f.close()
#     create_brain_npz()
    # sent_to_brain2()
#     # create_sentence_file()
#     # create_brain2txt()
#     # create_id2len()
# #
