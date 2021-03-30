import os
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
import sys
import time
sys.path.append(Proj_dir)
# print(sys.path)
import json
import torch
import scipy.io as scio
import numpy as np
# from src.com.util.data_format import normalization
from transformers import BertModel, BertTokenizer
# from src.com.model.run_auto_encoder import Autoencoder

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
# 这里调整模型
model_file = Proj_dir + '/benchmark/model_1e-05.bin'
dataset_name = "pereira"
participants = {'P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
                'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17'}

def get_sentence_embedding(sentence, option):
    # if YOU NEED [cls] and [seq] PRESENTATION：True
    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = bert_model(input_ids)

    all_layers_output = outputs[2]
    # list of torch.FloatTensor (one for the output of each layer + the output of the embeddings) of shape (batch_size, sequence_length, hidden_size): Hidden-states of the model at the output of each layer plus the initial embedding outputs.

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


def load_brain_data(dataset, user):
    if dataset == 'pereira':
        paticipants_1 = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
                         'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17', 'P01']
        paticipants_2 = ['M02', 'M04', 'M07', 'M08', 'M09', 'M14', 'M15', 'P01']
        paticipants_3 = ['M02', 'M03', 'M04', 'M07', 'M15', 'P01']
        a = []
        # dataset_path = "/Storage/ying/resources/pereira2018/'+user+'/data_180concepts_wordclouds.mat"
        if user in paticipants_1:
            exp1_path = '/Storage/ying/resources/pereira2018/' + user + '/data_180concepts_wordclouds.mat'
            exp1_data = scio.loadmat(exp1_path)
            examples = exp1_data['examples']
            print(user, examples.shape)
            ROI_path = Proj_dir+'/resource/' + user + '_roi.mat'
            data = scio.loadmat(ROI_path)
            # read roi index
            roi = data['index']

            for i in range((examples.shape[0])):
                # exp1_save_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/' + \
                #                  'pereira_' + user + '_exp1_' + str(i) + '.npy'
                b = []
                for index in roi[0]:
                    b.append(examples[i][index])
                # np.save(exp1_save_path, a)
                a.append(b[:46840])
            if user in paticipants_2:
                exp2_path = '/Storage/ying/resources/pereira2018/' + user + '/data_384sentences.mat'
                exp2_data = scio.loadmat(exp2_path)
                examples = exp2_data['examples_passagesentences']
                # b = []
                for i in range((examples.shape[0])):
                    # exp2_save_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/' + \
                    #                  'pereira_' + user + '_exp2_' + str(i) + '.npy'
                    # print(exp2_save_path)
                    b = []
                    for index in roi[0]:
                        b.append(examples[i][index])
                    # print(len(a))
                    # np.save(exp2_save_path, a)
                    a.append(b[:46840])
            if user in paticipants_3:
                exp3_path = '/Storage/ying/resources/pereira2018/' + user + '/data_243sentences.mat'
                exp3_data = scio.loadmat(exp3_path)
                # print(exp3_data.keys())
                examples = exp3_data['examples_passagesentences']
                for i in range((examples.shape[0])):
                    # print(i)
                    # exp3_save_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/' + \
                    #                  'pereira_' + user + '_exp3_' + str(i) + '.npy'
                    # print(exp3_save_path)
                    b = []
                    for index in roi[0]:
                        b.append(examples[i][index])
                    # print(len(a))
                    # np.save(exp3_save_path, a)
                    a.append(b[:46840])
    # if dataset == 'alice':

    return a


def calculate_corrcoef_pereira(dataset):
    import pandas as pd
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    sentence1_tsv_path = Proj_dir+'/resource/text_807.tsv'
    sentence2_tsv_path = Proj_dir+'/resource/text_564.tsv'
    sentence3_tsv_path = Proj_dir+'/resource/text_423.tsv'
    sentence4_tsv_path = Proj_dir+'/resource/text_180.tsv'
    # join exp1, 2and 3
    type1_participants = ['P01', 'M02', 'M04', 'M07', 'M15']
    # join exp1 and 3
    type2_participants = ['M08', 'M09', 'M14']
    # join exp1 and 3
    type3_participants = ['M03']
    # only join exp1
    type4_participants = ['M01', 'M05', 'M06', 'M10', 'M13', 'M16', 'M17']
    corr_user = {}
    for user in participants:
        #     file_spilt = file.replace('.npy', '').split('_')
        brain_data = load_brain_data(dataset, user)
        #     user = file_spilt[1]
        #     index = int(file_spilt[3])

        train_X = np.array(brain_data)
        train_X, _, _ = normalization(train_X)
        train_X = torch.Tensor(train_X)
        train_X = train_X.to(device, dtype=torch.float)
        y, z = pre_trained_model(train_X)
        # encoded_imgs, _, _ = normalization(encoded_imgs)
        # z = tf.reshape(encoded_imgs, [-1, 1024])
        # norm_shape = train_X[0].shape
        corr_list = []
        # feature_shape = z[0].shape[0]
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
        for index in range((z.shape[0])):
            sentence = sentence_df.loc[index]['text']
            sent_embeddings, t = get_sentence_embedding(sentence, "last_layer")
            # z[index, :].cpu().detach().numpy(), t.cpu().detach().numpy()
            corr = np.corrcoef(z[index, :].cpu().detach().numpy(), t.cpu().detach().numpy())[0, 1]
            corr_list.append(corr)

            # npz_filename = dataset_name + '_' + user + '_' + str(index) + '.npz'
            # norm_data['norm'] = {'data': train_X[index].cpu().data.tolist(),
            #                      'shape': list(norm_shape),
            #                      'type': None,
            #                      'kind': None}
            # feature_data['features'] = {'data': z[index].cpu().data.numpy().tolist(),
            #                             'shape': [feature_shape],
            #                             'type': None,
            #                             'kind': None}
            # np.savez(feature_path + npz_filename, features=feature_data['features'])
            # np.savez(norm_path + npz_filename, features=norm_data['norm'])

            # print(k[1])
        corr_user[user] = corr_list
    with open(dataset+f'_corr_user.json', 'w') as f:
            json.dump(corr_user, f)
    # json.dump(corr_user, dataset+f'_corr_user.json')

def load_text_embedding_list():
    import scipy.io as io
    data = io.loadmat('text_embedding.mat')
    # print("text_embedding shape:", data['result'].shape)
    # print(data['text'])
    return data['result']

def calculate_corrcoef_alice(dataset):
    if dataset == 'alice':
        corr_user = {}
        participants_alice = {'18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37',
                              '39', '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53'}

        import tensorflow.keras as keras
        from src.com.util.data_format import noramlization, generate_array_from_file
        import tensorflow as tf
        import numpy as np
        from tensorflow.keras.models import load_model
        IMAGE_HEIGHT = 72
        IMAGE_WIDTH = 96
        IMAGE_DEPTH = 64
        text_embedding = load_text_embedding_list()
        model = load_model('model/autoencoder_mri_kf.h5')
        pretrained_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer("latent").output)  # 你创建新的模型
        for sub_id in participants_alice:
            corr_list = []
            tf_path = '/Storage/ying/resources/BrainBertTorch/dataset/alice/tf_data/test_data_vol_' + sub_id + '.tfrecords'
            test, label = generate_array_from_file(tf_path, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)
            train_X = np.array(test).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])
            train_X, _, _ = noramlization(train_X)
            # a.append(train_X)
            encoded_imgs = pretrained_model.predict(train_X)
            encoded_imgs, _, _ = noramlization(encoded_imgs)
            z = tf.reshape(encoded_imgs, [-1, 1024])
            for index in range((z.shape[0])):
                corr = np.corrcoef(z[index, :].cpu().numpy(), text_embedding[index])[0, 1]
                print(corr)
                corr_list.append(corr)
            corr_user[sub_id] = corr_list
        with open(dataset + f'_corr_user.json', 'w') as f:
                json.dump(corr_user, f)




if __name__ == "__main__":
    start = time.perf_counter()

    # dataset = 'pereira'
    # feature_path = '/Storage/ying/resources/BrainBertTorch/dataset/' + dataset + '/features/'
    # user = 'P01'
    # npz_filename = dataset_name + '_' + user + '_' + str(0) + '.npz'
    # # participants = {'P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
    # #                 'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17'}
    # features = np.load(feature_path+npz_filename)
    # print(features['data'])
    # calculate_corrcoef_pereira('pereira')
    calculate_corrcoef_alice('alice')
    end = time.perf_counter()
    time_cost = str((end - start) / 60)
    print("time-cost:", time_cost)

