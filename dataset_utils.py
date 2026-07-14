# Author: Yu Chen
# Data: 2025/09/21
# Email: yu2000.chen@connect.polyu.hk



import random
from operator import index

import torch
from sympy.benchmarks.bench_meijerint import normal
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import os
import glob


def create_training_and_test_dataset(origin_training_set_address, training_set_file_name, test_set_file_name, test_set_rate, hard_rate, noisy_rate):
    """
    通过写地址的方式，写数据集，噪声样本指的是不同类别之间
    hard 数据指的是同一故障类型很难区分（这个是指的是同类型故障之间的）
    noisy 数据指的是不同故障类型之间的噪声样本（肉眼看出，是因为工作人员不小心而划分错误）
    origin_training_set_address: 原始数据集的地址
    training_set_file_name: 训练数据集的名称
    test_set_file_name: 测试数据集的名称
    test_set_rate: 测试集样本的比例
    hard_rate: hard样本的比率
    noisy_rate： noisy样本的比率

    创建的地址csv， 第一列是 Gain 的地址，第二列是 Phase 的地址，第三列是实际使用的标签，第四列是真实标签，第五列为什么样本(0, 正常样本)
    """

    # 数据初始化
    def check_and_delete_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"previous {file_path} is been deleted")
        else:
            print(f"{file_path} does not exist")

    check_and_delete_file(training_set_file_name)
    check_and_delete_file(test_set_file_name)
    hard_sample_number = 0
    noisy_sample_number = 0
    normal_sample_number = 0
    training_dataset_number = 0
    test_dataset_number = 0
    total_number_sample = 0
    training_dataset_df = pd.DataFrame()
    test_dataset_df = pd.DataFrame()
    test_sample_per_turn = int(1/test_set_rate)


    label_dic = {"#1GFSC":0, "#2GFSC":1,  "#3GFSC":2, "#2-#3IDSC":3, "#2-#4IDSC":4,  "#3-#4IDSC":5,  "Normal": 6}
    fault_type_list = os.listdir(origin_training_set_address)
    for fault_type in fault_type_list:
        fault_type_path = os.path.join(origin_training_set_address, fault_type)
        sample_file_name_list =  os.listdir(fault_type_path)
        for sample_file_name in sample_file_name_list:
            label_name, remaining_str = sample_file_name.split("_", 1)
            if "Gain" in remaining_str:
                total_number_sample += 1
                print(f"total_number_sample: {total_number_sample}")
                sample_Gain_file_path = os.path.join(fault_type_path, sample_file_name)
                sample_Phase_file_path = sample_Gain_file_path.replace("Gain", "Phase")
                real_label = label_dic[label_name]
                # 准备测试集数据
                if total_number_sample % test_sample_per_turn == 0:
                    test_dataset_number += 1

                    test_dataset_temp_df = pd.DataFrame({"Gain_address": [sample_Gain_file_path],
                                              "Phase_address": [sample_Phase_file_path],
                                              "Read_label":[real_label]})
                    test_dataset_df = pd.concat([test_dataset_df, test_dataset_temp_df])
                # 准备训练集数据
                else:
                    training_dataset_number += 1
                    random_prob = np.random.rand()
                    # noise 样本 (随机抽取另外几个样本标签，这个样本，我们认为) 这里的hard和noise的标签都是错的，和后面的hard不太一样
                    if random_prob <= noisy_rate:
                        noisy_sample_number += 1
                        label_list = [0, 1, 2, 3, 4, 5, 6]

                        # method 1
                        # GFSC_list = [0, 1, 2]
                        # IDSC_list = [3, 4, 5]
                        # if real_label in GFSC_list:
                        #     for single_item in GFSC_list:
                        #         label_list.remove(single_item)
                        # elif real_label in IDSC_list:
                        #     for single_item in IDSC_list:
                        #         label_list.remove(single_item)
                        # elif real_label ==6:
                        #     label_list.remove(6)

                        # method 2
                        label_list.remove(real_label)

                        noise_label = random.choice(label_list)
                        training_dataset_temp_df = pd.DataFrame({"Gain_address": [sample_Gain_file_path],
                                                  "Phase_address": [sample_Phase_file_path],
                                                  "Read_label":[real_label],
                                                  "Wrong_label":[noise_label],
                                                  "Wrong_label_type":["noise"]})

                    # hard 样本 (在同一类型中随机抽取,这一类型不包括正常的，因为正常和其他几种样本都不像)
                    elif (noisy_rate<random_prob) and (random_prob<=noisy_rate+hard_rate) and (real_label != 6):
                        hard_sample_number += 1
                        GFSC_list = [0, 1, 2]
                        IDSC_list = [3, 4, 5]
                        if real_label in GFSC_list:
                            GFSC_list.remove(real_label)
                            hard_label = random.choice(GFSC_list)
                        else:
                            IDSC_list.remove(real_label)
                            hard_label = random.choice(IDSC_list)
                        training_dataset_temp_df = pd.DataFrame({"Gain_address": [sample_Gain_file_path],
                                                                 "Phase_address": [sample_Phase_file_path],
                                                                 "Read_label": [real_label],
                                                                 "Wrong_label": [hard_label],
                                                                 "Wrong_label_type": ["hard"]})
                    # 正常样本 没有任何问题
                    else:
                        normal_sample_number += 1
                        training_dataset_temp_df = pd.DataFrame({"Gain_address": [sample_Gain_file_path],
                                                                 "Phase_address": [sample_Phase_file_path],
                                                                 "Read_label": [real_label],
                                                                 "Wrong_label": [real_label],
                                                                 "Wrong_label_type": ["normal"]})
                    training_dataset_df = pd.concat([training_dataset_df, training_dataset_temp_df])

    # 在此处shuffle数据 并且写出相应的excel
    test_dataset_df = test_dataset_df.sample(frac=1, random_state=None)
    training_dataset_df = training_dataset_df.sample(frac=1, random_state=None)

    with pd.ExcelWriter(test_set_file_name, engine='openpyxl') as writer:
        test_dataset_df.to_excel(writer, sheet_name='Dataset_information', index=False)
    with pd.ExcelWriter(training_set_file_name, engine='openpyxl') as writer:
        training_dataset_df.to_excel(writer, sheet_name='Dataset_information', index=False)

    # print  数据集信息
    print("----------------------Dataset information----------------")
    print(f"total_number_sample: {total_number_sample}")
    print(f"training_dataset_number: {training_dataset_number}")
    print(f"normal_sample_number: {normal_sample_number}")
    print(f"noisy_sample_number: {noisy_sample_number}")
    print(f"hard_sample_number: {hard_sample_number}")
    print(f"test_dataset_number: {test_dataset_number}")
    print("---------------End of Dataset information----------------")


def use_selected_simple_samples_to_build_a_artifical_dataset(training_set_file_name, noisy_rate):
    """
    Create dataset for auxiliary model, only including noise and simple samples.
    training_set_file_name: excel name
    wrong_label_rate: wrong label rate for auxiliary model
    """
    # Read related files
    simple_sample_idx = pd.read_excel(training_set_file_name, sheet_name="Simple_sample_index", header=0)
    Dataset_information = pd.read_excel(training_set_file_name, sheet_name="Dataset_information", header=0)
    Dataset_information_for_artifical_dataset = pd.DataFrame([])
    simple_sample_idx = simple_sample_idx.iloc[:, 0].tolist()

    index_number = 0
    normal_sample_number = 0
    noisy_sample_number = 0
    for index in simple_sample_idx:
        index_number += 1
        print(f"{index_number}/ {len(simple_sample_idx)}:Please wait for the dataset to be built....")
        random_prob = np.random.rand()
        Artifical_dataset_label = int(Dataset_information.iloc[index]["Wrong_label"])
        Artifical_dataset_real_label = Artifical_dataset_label
        Artifical_dataset_label_type = "normal"
        ## make noise sample
        if random_prob<=noisy_rate:
            noisy_sample_number += 1
            label_list = [0, 1, 2, 3, 4, 5, 6]
            # method 1
            # GFSC_list = [0, 1, 2]
            # IDSC_list = [3, 4, 5]
            #
            # if  Auxiliary_model_label in GFSC_list:
            #     for single_item in GFSC_list:
            #         label_list.remove(single_item)
            # elif Auxiliary_model_label in IDSC_list:
            #     for single_item in IDSC_list:
            #         label_list.remove(single_item)
            # elif Auxiliary_model_label==6:
            #     label_list.remove(6)

            # method 2
            label_list.remove(Artifical_dataset_label)
            Artifical_dataset_label = random.choice(label_list)
            Artifical_dataset_label_type = "noise"

        # Normal sample
        else:
            normal_sample_number += 1

        Dataset_information_for_artifical_dataset_temp = pd.DataFrame({"Sample_ID": [index],
                             "Gain_address": Dataset_information.iloc[index]["Gain_address"],
                             "Phase_address": Dataset_information.iloc[index]["Phase_address"],
                             "Read_label":Dataset_information.iloc[index]["Read_label"],
                             "Wrong_label":Dataset_information.iloc[index]["Wrong_label"],
                             "Wrong_label_type":Dataset_information.iloc[index]["Wrong_label_type"],
                             "Artifical_dataset_real_label":[Artifical_dataset_real_label],
                             "Artifical_dataset_label":[Artifical_dataset_label],
                             "Artifical_dataset_label_type":[Artifical_dataset_label_type]})
        Dataset_information_for_artifical_dataset = pd.concat([Dataset_information_for_artifical_dataset, Dataset_information_for_artifical_dataset_temp])
    ## save data into excel file
    with pd.ExcelWriter(training_set_file_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        Dataset_information_for_artifical_dataset.to_excel(writer, sheet_name="Dataset_information_for_AM", index=False)


    print("------------------Artifical dataset information-------------------------")
    print(f"normal sample number:{normal_sample_number}")
    print(f"noisy sample number:{noisy_sample_number}")
    print("--------------------End of artifical dataset information----------------")


class Auxiliary_model_training_dataset_or_artifical_dataset(Dataset):
    def __init__(self, training_set_file_name, whether_is_artifical_dataset=False, resize=128):
        super(Auxiliary_model_training_dataset_or_artifical_dataset, self).__init__()
        self.resize = resize
        self.training_set_file_name = training_set_file_name
        self.whether_is_artifical_dataset = whether_is_artifical_dataset

        if self.whether_is_artifical_dataset == False:
            self.Auxiliary_model_label_dic = {"noise":0, "hard":1}
            self.loss_data, self.label_data = self.load_excel_file_for_auxiliary_model(training_set_file_name)
            self.number_of_sample = self.label_data.shape[0]
            self.loss_data_np = np.array(self.loss_data)
            self.label_data_np = np.array(self.label_data)

        else:
            self.Gain_address, self.Phase_address, self.Artifical_dataset_real_label, self.Artifical_dataset_label, self.Artifical_dataset_label_type = self.load_excel_file_for_artifical_dataset(training_set_file_name)
            self.number_of_sample = len(self.Gain_address)

    def load_excel_file_for_artifical_dataset(self, filename):
        data = pd.read_excel(filename, sheet_name="Dataset_information_for_AM", header=0)
        Gain_address = data["Gain_address"].tolist()
        Phase_address = data["Phase_address"].tolist()
        Artifical_dataset_real_label = data["Artifical_dataset_real_label"].tolist()
        Artifical_dataset_label = data["Artifical_dataset_label"].tolist()
        Artifical_dataset_label_type = data["Artifical_dataset_label_type"].tolist()
        return Gain_address, Phase_address, Artifical_dataset_real_label, Artifical_dataset_label, Artifical_dataset_label_type

    def load_excel_file_for_auxiliary_model(self, filename):
        Auxiliary_model_dataset = pd.read_excel(filename, sheet_name="Auxiliary_model_dataset", header=0)
        Loss_function_for_AM = pd.read_excel(filename, sheet_name="Loss_function_for_AM", header=0)
        Loss_function_for_AM = np.array(Loss_function_for_AM)
        sample_index_list = Auxiliary_model_dataset["Auxiliary_model_data_index"].tolist()
        Auxiliary_model_label_type_list = Auxiliary_model_dataset["Auxiliary_model_label"].tolist()
        loss_data = np.zeros((len(sample_index_list), Loss_function_for_AM.shape[1]))
        label_data = np.zeros(len(sample_index_list))
        sample_number = 0
        for sample_index in sample_index_list:
            loss_data[sample_number, :] = Loss_function_for_AM[sample_index, :]
            label_data[sample_number] =self.Auxiliary_model_label_dic[Auxiliary_model_label_type_list[sample_number]]
            sample_number += 1
        return loss_data, label_data

    def __len__(self):
        if self.whether_is_artifical_dataset == False:
            return self.label_data.shape[0]
        else:
            return len(self.Gain_address)

    def __getitem__(self, item):
        if self.whether_is_artifical_dataset == False:
            return torch.tensor(self.loss_data[item, :], dtype=torch.float32), torch.tensor(self.label_data[item], dtype=torch.long)
        else:
            Gain_address, Phase_address, Auxiliary_model_real_label, Auxiliary_model_label, Auxiliary_model_label_type = self.Gain_address[item], self.Phase_address[item], self.Artifical_dataset_real_label[item], self.Artifical_dataset_label[item], self.Artifical_dataset_label_type[item]
            Auxiliary_model_label = torch.tensor(Auxiliary_model_label, dtype=torch.long)

            tf = transforms.Compose([
                lambda x: Image.open(x),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
            ])
            Gain, Phase = tf(Gain_address), tf(Phase_address)
            return Gain, Phase, Auxiliary_model_label, Auxiliary_model_real_label, Auxiliary_model_label_type


class SFRA_dataset(Dataset):
    def __init__(self, whether_is_training, training_set_file_name, test_set_file_name, resize):
        super(SFRA_dataset, self).__init__()
        self.whether_is_training = whether_is_training
        if self.whether_is_training:
            self.root = training_set_file_name
            self.Gain_address_list, self.Phase_address_list, self.Real_label_list, self.Wrong_label_list, self.Wrong_label_type_list = self.load_csv(whether_is_training, self.root)
        else:
            self.root = test_set_file_name
            self.Gain_address_list, self.Phase_address_list, self.Real_label_list = self.load_csv(whether_is_training, self.root)
        self.whether_is_training = whether_is_training
        self.resize = resize
        self.number_of_sample = len(self.Gain_address_list)

    def load_csv(self, whether_is_training, filename):
        data = pd.read_excel(filename, sheet_name="Dataset_information")
        Gain_address_list = data.iloc[:, 0].tolist()
        Phase_address_list = data.iloc[:, 1].tolist()
        Real_label_list = data.iloc[:, 2].tolist()
        if whether_is_training:
            Wrong_label_list = data.iloc[:, 3].tolist()
            Wrong_label_type_list = data.iloc[:, 4].tolist()
            return Gain_address_list, Phase_address_list, Real_label_list, Wrong_label_list, Wrong_label_type_list
        return Gain_address_list, Phase_address_list, Real_label_list

    def __len__(self):
        return len(self.Gain_address_list)

    def __getitem__(self, item):
        if self.whether_is_training:
            Gain_address, Phase_address, Real_label, Wrong_label, Wrong_label_type= self.Gain_address_list[item], self.Phase_address_list[item], self.Real_label_list[item], self.Wrong_label_list[item], self.Wrong_label_type_list[item]
            label = torch.tensor(Wrong_label)
        else:
            Gain_address, Phase_address, Real_label  = self.Gain_address_list[item], self.Phase_address_list[item], self.Real_label_list[item]
            label = torch.tensor(Real_label)

        tf = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
        ])
        Gain, Phase = tf(Gain_address), tf(Phase_address)

        if self.whether_is_training:
            return Gain, Phase, label, Real_label, Wrong_label_type
        else:
            return Gain, Phase, label




class label_or_unlabel_dataset(Dataset):
    def __init__(self, whether_dataset_is_label, training_set_file_name, resize=128):
        super(label_or_unlabel_dataset, self).__init__()
        # noise: no label for VAT, hard: label for VAT
        self.hard_and_noise_label_dict = {"noise":0, "hard":1}
        self.whether_dataset_is_label = whether_dataset_is_label
        self.training_set_file_name = training_set_file_name
        self.resize = resize

        hard_and_noise_sample_index = pd.read_excel(self.training_set_file_name, sheet_name="hard_and_noise_sample_index", header=0)
        hard_and_noise_sample_ID = np.array(hard_and_noise_sample_index["Sample_ID"])
        hard_and_noise_sample_type = np.array(hard_and_noise_sample_index["hard(1)_and_noise(0)_predict"])
        noise_label_index = np.where(hard_and_noise_sample_type==0)
        noise_sample_index = hard_and_noise_sample_ID[noise_label_index]
        hard_label_index = np.where(hard_and_noise_sample_type == 1)
        hard_sample_index = hard_and_noise_sample_ID[hard_label_index]

        simple_sample_index = pd.read_excel(self.training_set_file_name, sheet_name="Simple_sample_index", header=0)
        simple_sample_index = np.array(simple_sample_index["Sample_ID"])
        self.label_dataset_index = np.hstack((simple_sample_index, hard_sample_index))
        self.unlabel_dataset_index = noise_sample_index
        self.label_dataset_number = self.label_dataset_index.shape[0]
        self.unlabel_dataset_number = self.unlabel_dataset_index.shape[0]
        self.Gain_address_list, self.Phase_address_list, self.Real_label_list, self.Wrong_label_list, self.Wrong_label_type_list = self.load_excel(self.training_set_file_name)

    def load_excel(self, filename):
        data = pd.read_excel(filename, sheet_name="Dataset_information", header=0)
        Gain_address_list = data.iloc[:, 0].tolist()
        Phase_address_list = data.iloc[:, 1].tolist()
        Real_label_list = data.iloc[:, 2].tolist()
        Wrong_label_list = data.iloc[:, 3].tolist()
        Wrong_label_type_list = data.iloc[:, 4].tolist()
        return Gain_address_list, Phase_address_list, Real_label_list, Wrong_label_list, Wrong_label_type_list


    def __len__(self):
        if self.whether_dataset_is_label:
            return self.label_dataset_index.shape[0]
        else:
            return self.unlabel_dataset_index.shape[0]

    def __getitem__(self, item):

        tf = transforms.Compose([
            lambda x: Image.open(x),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
        ])

        if self.whether_dataset_is_label:
            number = self.label_dataset_index[item]
            Gain_address, Phase_address, Wrong_label, Real_label = self.Gain_address_list[number], self.Phase_address_list[number], self.Wrong_label_list[number], self.Real_label_list[number]
            Gain, Phase = tf(Gain_address), tf(Phase_address)
            Wrong_label = torch.tensor(Wrong_label, dtype=torch.long)
            return Gain, Phase, Wrong_label
        else:
            number = self.unlabel_dataset_index[item]
            Gain_address, Phase_address = self.Gain_address_list[number], self.Phase_address_list[number]
            Gain, Phase = tf(Gain_address), tf(Phase_address)
            return Gain, Phase



def test_function_1(whether_create_dataset=False):
    if whether_create_dataset:
        create_training_and_test_dataset(origin_training_set_address="Training_dataset_3",
                                         training_set_file_name="./data_temp_log/Training_dataset_3.xlsx",
                                         test_set_file_name="./data_temp_log/test_dataset_3.xlsx",
                                         test_set_rate=0.05, hard_rate=0.05, noisy_rate=0.05)

    data = SFRA_dataset(whether_is_training=True,
                        training_set_file_name="./data_temp_log/Training_dataset_3.xlsx",
                        test_set_file_name="./data_temp_log/test_dataset_3.xlsx",
                        resize=128)

    print(data.number_of_sample)
    a = DataLoader(data, batch_size=1, num_workers=0)
    # write = SummaryWriter("logs")
    for i, (Gain, Phase, label, Real_label, Wrong_label_type) in enumerate(a):
        print(label)
        print(Real_label)
        print(Wrong_label_type)
        break
    #     write.add_images("Gain", Gain, global_step=i)
    #     write.add_images("Phase", Phase, global_step=i)
    # write.close()

if __name__ == '__main__':
    #test_function_1(True)
    data_set = label_or_unlabel_dataset(whether_dataset_is_label=True, training_set_file_name="./data_temp_log/Training_dataset_3.xlsx")
    number = 0
    index = 0
    for (_, _, a, b) in data_set:
        if a != b:
            print(a, b)
            print(index)
        if a ==b:
            number += 1
        index+=1

    a = 0
