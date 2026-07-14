# -*- coding: utf-8 -*-
# Author:Yu Chen
# Data: 2025/9/22 12:06
# Email: yu2000.chen@connect.polyu.hk

import torchvision
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model_utils import initialize_weights, Normal_encoder, Auxiliary_model, Normal_encoder_1
from dataset_utils import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.Image
import io

def intital_setup_and_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train_backbone_model_with_dynamics(training_dataset, test_dataset, training_set_file_name,
                                       device, logs_dir, loss_function_sheet_name="Loss_function",
                                       epochs=100, batch_size=32, lr=0.001, record_losses=True,
                                       whether_use_test_dataset=True, whether_use_noise_testdat_to_evaluate=True,
                                       noise_dataset_name="Test_dataset_8"):
    """
    Train backbone and return model + loss dynamics (n_samples x epochs array).
    - records per-sample losses over epochs (for dynamics in Stage I).
    training_dataset
    test_dataset
    epochs
    batch_size
    lr: learning rate
    record_losses: whether to record losses over epochs (for dynamics in Stage I)
    device
    """

    # model setting
    encoder = Normal_encoder_1(128, 1, 2)
    encoder = encoder.to(device)
    initialize_weights(encoder)
    # print(encoder)

    # training setting
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=lr)

    # dataset setting
    training_dataset_loader = DataLoader(training_dataset, batch_size=batch_size, num_workers=0)
    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    # other setting
    recorded_losses = np.zeros((training_dataset.number_of_sample, epochs)) if record_losses else None
    write = SummaryWriter(logs_dir)

    if whether_use_noise_testdat_to_evaluate:
        noise_dataset_name = noise_dataset_name
        noise_test_dataset_1 = SFRA_dataset(whether_is_training=False,
                                    training_set_file_name=f"./data_temp_log/{noise_dataset_name}.xlsx",
                                    test_set_file_name=f"./data_temp_log/{noise_dataset_name}_db10.xlsx",
                                    resize=128)

        noise_test_dataset_2 = SFRA_dataset(whether_is_training=False,
                                            training_set_file_name=f"./data_temp_log/{noise_dataset_name}.xlsx",
                                            test_set_file_name=f"./data_temp_log/{noise_dataset_name}_db20.xlsx",
                                            resize=128)
        noise_test_dataset_3 = SFRA_dataset(whether_is_training=False,
                                            training_set_file_name=f"./data_temp_log/{noise_dataset_name}.xlsx",
                                            test_set_file_name=f"./data_temp_log/{noise_dataset_name}_db30.xlsx",
                                            resize=128)
        noise_test_dataset_loader_1 = DataLoader(noise_test_dataset_1, batch_size=batch_size, num_workers=0)
        noise_test_dataset_loader_2 = DataLoader(noise_test_dataset_2, batch_size=batch_size, num_workers=0)
        noise_test_dataset_loader_3 = DataLoader(noise_test_dataset_3, batch_size=batch_size, num_workers=0)


    for epoch in range(epochs):

        # training model
        encoder.train()
        epoch_loss = 0
        total_training_accuracy = 0
        for batch_idx, (Gain, Phase, label, _, _) in enumerate(training_dataset_loader):
            Gain, Phase, label = Gain.to(device), Phase.to(device), label.to(device)
            output = encoder(Gain, Phase)
            loss = loss_function(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            accuracy = (output.argmax(1) == label).sum()
            total_training_accuracy += accuracy

        print(f"{epoch+1}/{epochs}, training accuracy:{total_training_accuracy / training_dataset.number_of_sample}")
        write.add_scalar("training_loss", epoch_loss, epoch)
        write.add_scalar("training_accuracy", total_training_accuracy/training_dataset.number_of_sample, epoch)

        # Record per-sample losses (forward pass on full train set)
        if record_losses:
            encoder.eval()
            with torch.no_grad():
                for batch_idx, (Gain, Phase, label, _, _) in enumerate(training_dataset_loader):
                    Gain, Phase, label = Gain.to(device), Phase.to(device), label.to(device)
                    output = encoder(Gain, Phase)
                    recorded_loss_function = nn.CrossEntropyLoss(reduction='none')
                    per_sample_loss = recorded_loss_function(output, label).cpu().detach().numpy()
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, training_dataset.number_of_sample)
                    recorded_losses[start_idx:end_idx, epoch] = per_sample_loss

        if whether_use_test_dataset:
            # evaluate model in test dataset
            total_test_accuracy = 0
            encoder.eval()
            with torch.no_grad():
                for (Gain, Phase, label) in test_dataset_loader:
                    Gain, Phase, label = Gain.to(device), Phase.to(device), label.to(device)
                    output = encoder(Gain, Phase)
                    accuracy = (output.argmax(1) == label).sum()
                    total_test_accuracy += accuracy

                print(f"{epoch+1}/{epochs}, test accuracy:{total_test_accuracy / test_dataset.number_of_sample}")
                write.add_scalar("test_accuracy", total_test_accuracy / test_dataset.number_of_sample, epoch)

            if whether_use_noise_testdat_to_evaluate:
                with torch.no_grad():
                    for i, noise_test_dataset_loader in enumerate([noise_test_dataset_loader_1, noise_test_dataset_loader_2, noise_test_dataset_loader_3],start=1):
                        all_preds = []
                        all_labels = []
                        total_test_accuracy = 0
                        for (Gain, Phase, label) in noise_test_dataset_loader:
                            Gain, Phase, label = Gain.to(device), Phase.to(device), label.to(device)
                            output = encoder(Gain, Phase)
                            accuracy = (output.argmax(1) == label).sum()
                            total_test_accuracy += accuracy
                            preds = output.argmax(1).cpu().numpy()
                            labels = label.cpu().numpy()
                            all_preds.extend(preds)
                            all_labels.extend(labels)

                        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                        print(f"Epoch:{epoch + 1}/{epochs}, Noise test dataset {i} accuracy_before_VAT:{total_test_accuracy / noise_test_dataset_loader.dataset.number_of_sample}")
                        print(f"Epoch:{epoch + 1}/{epochs}, Noise test dataset {i} precision_before_VAT:{precision}")
                        print(f"Epoch:{epoch + 1}/{epochs}, Noise test dataset {i} recall_before_VAT:{recall}")
                        print(f"Epoch:{epoch + 1}/{epochs}, Noise test dataset {i} f1_before_VAT:{f1}")
                        write.add_scalar(f"noise_test_dataset_{i}_accuracy_before_VAT",total_test_accuracy / noise_test_dataset_loader.dataset.number_of_sample, epoch)
                        write.add_scalar(f"noise_test_dataset_{i}_precision_before_VAT", precision, epoch)
                        write.add_scalar(f"noise_test_dataset_{i}_recall_before_VAT", recall, epoch)
                        write.add_scalar(f"noise_test_dataset_{i}_f1_before_VAT", f1, epoch)


    ## save loss data
    df_losses = pd.DataFrame(recorded_losses, columns=[f'Epoch_{i + 1}' for i in range(recorded_losses.shape[1])])
    with pd.ExcelWriter(training_set_file_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_losses.to_excel(writer, sheet_name=loss_function_sheet_name, index=False)


def train_auxiliary_model_based_on_LSTM(Auxiliary_model_dataset, loss_sequence_size, device, logs_dir, epochs=100, batch_size=32, lr=0.001):
    """
    train auxiliary model
    """
    # initial setting
    AM = Auxiliary_model(loss_sequence_size=loss_sequence_size)
    AM = AM.to(device)
    initialize_weights(AM)
    Auxiliary_model_dataset_loader = DataLoader(Auxiliary_model_dataset, batch_size=batch_size, num_workers=0)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(AM.parameters(), lr=lr)
    write = SummaryWriter(logs_dir)
    best_loss = 100
    best_model_path = os.path.join(logs_dir, 'best_aux_model.pth')

    for epoch in range(epochs):
        AM.train()
        epoch_loss = 0
        total_training_accuracy = 0
        for batch_idx, (loss_data, label) in enumerate(Auxiliary_model_dataset_loader):
            loss_data, label = loss_data.to(device), label.to(device)
            output = AM(loss_data)
            loss = loss_function(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            accuracy = (output.argmax(1) == label).sum()
            total_training_accuracy += accuracy

        acc = total_training_accuracy / Auxiliary_model_dataset.number_of_sample
        print(f"{epoch + 1}/{epochs}, training accuracy:{acc}")
        print(f"{epoch + 1}/{epochs}, training loss:{epoch_loss}")
        write.add_scalar("training_loss", epoch_loss, epoch)
        write.add_scalar("training_accuracy", acc, epoch)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(AM.state_dict(), best_model_path)

    return AM


def train_auxiliary_model_based_on_SVM_or_Randomforest(Auxiliary_model_dataset, model_type="SVM"):
    au_loss_data = Auxiliary_model_dataset.loss_data_np
    au_label_data = Auxiliary_model_dataset.label_data_np
    if model_type == "SVM":
        aux_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        aux_model.fit(au_loss_data, au_label_data)
    elif model_type == "RandomForest":
        aux_model = RandomForestClassifier(n_estimators=100, random_state=42)
        aux_model.fit(au_loss_data, au_label_data)
    aux_acc = accuracy_score(au_label_data, aux_model.predict(au_loss_data))
    print(f"Auxiliary model accuracy ({model_type}): {aux_acc}")
    return aux_model


def Virtual_adversarial_training_for_backbone_model(label_dataset, unlabel_dataset, test_dataset, device, logs_dir, epochs=100, batch_size=32, lr=0.001, alpha=1.0, whether_use_noise_testdat_to_evaluate=False, noise_dataset_name="Test_dataset_8"):

    def kl_divergence(p, q):
        epsilon = 1e-10
        assert p.shape == q.shape, f"Shape mismatch: p {p.shape}, q {q.shape}"
        p = torch.softmax(p, dim=1)
        q = torch.softmax(q, dim=1)
        return torch.sum(p * (torch.log(p + epsilon) - torch.log(q + epsilon)), dim=1)

    def virtual_adversarial_perturbation(Gain, Phase, model, epsilon=0.3):
        Gain = Gain.detach()
        Phase = Phase.detach()
        p = model(Gain, Phase)
        delta = torch.randn_like(Gain) * 0.1
        delta.requires_grad_(True)
        for _ in range(3):
            q = model(Gain + delta, Phase + delta)
            div = kl_divergence(p, q)
            grad = torch.autograd.grad(div.sum(), delta, create_graph=True)[0]
            delta = delta + epsilon * grad / (torch.norm(grad) + 1e-8)
            delta = delta * min(1., epsilon / (torch.norm(delta) + 1e-8))
            delta.requires_grad_(True)
        return delta.detach()

    def vat_loss(Gain, Phase, model, epsilon=0.3):
        r_vadv = virtual_adversarial_perturbation(Gain, Phase, model, epsilon)
        p_orig = model(Gain, Phase)
        p_pert = model(Gain + r_vadv, Phase + r_vadv)
        return kl_divergence(p_orig, p_pert).mean()


    model = Normal_encoder(128, 1, 2)
    initialize_weights(model)
    model = model.to(device)
    write = SummaryWriter(logs_dir)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    label_dataset_loader = DataLoader(label_dataset, batch_size=batch_size, num_workers=0)
    unlabel_dataset_loader = DataLoader(unlabel_dataset, batch_size=batch_size, num_workers=0)
    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    if whether_use_noise_testdat_to_evaluate:

        noise_test_dataset_1 = SFRA_dataset(whether_is_training=False,
                                    training_set_file_name=f"./data_temp_log/{noise_dataset_name}.xlsx",
                                    test_set_file_name=f"./data_temp_log/{noise_dataset_name}_db10.xlsx",
                                    resize=128)

        noise_test_dataset_2 = SFRA_dataset(whether_is_training=False,
                                            training_set_file_name=f"./data_temp_log/{noise_dataset_name}.xlsx",
                                            test_set_file_name=f"./data_temp_log/{noise_dataset_name}_db20.xlsx",
                                            resize=128)
        noise_test_dataset_3 = SFRA_dataset(whether_is_training=False,
                                            training_set_file_name=f"./data_temp_log/{noise_dataset_name}.xlsx",
                                            test_set_file_name=f"./data_temp_log/{noise_dataset_name}_db30.xlsx",
                                            resize=128)
        noise_test_dataset_loader_1 = DataLoader(noise_test_dataset_1, batch_size=batch_size, num_workers=0)
        noise_test_dataset_loader_2 = DataLoader(noise_test_dataset_2, batch_size=batch_size, num_workers=0)
        noise_test_dataset_loader_3 = DataLoader(noise_test_dataset_3, batch_size=batch_size, num_workers=0)


    criterion = nn.CrossEntropyLoss()
    best_loss = 9999999
    best_model_path = os.path.join(logs_dir, 'best_model.pth')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_training_correct = 0
        label_iter = iter(label_dataset_loader)
        unlabel_iter = iter(unlabel_dataset_loader)
        for _ in range(len(label_dataset_loader)):
            # label dataset Clf loss
            try:
                batch_Gain, batch_Phase, batch_y = next(label_iter)
            except StopIteration:
                break
            batch_Gain, batch_Phase, batch_y = batch_Gain.to(device), batch_Phase.to(device), batch_y.to(device)
            outputs = model(batch_Gain, batch_Phase)
            clf_loss = criterion(outputs, batch_y)
            vat_l = vat_loss(batch_Gain, batch_Phase, model)
            accuracy = (outputs.argmax(1) == batch_y).sum()
            total_training_correct += accuracy

            # unlabel dataset VAT loss
            try:
                unlabeled_Gain, unlabeled_Phase = next(unlabel_iter)
                unlabeled_Gain, unlabeled_Phase = unlabeled_Gain.to(device), unlabeled_Phase.to(device)
                vat_ul = vat_loss(unlabeled_Gain, unlabeled_Phase, model)
            except StopIteration:
                vat_ul = 0

            # 联合优化
            loss = clf_loss + alpha * (vat_ul + vat_l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # evaluate training accuracy in training dataset
        training_accuracy = total_training_correct / label_dataset.label_dataset_number
        print(f"Epoch {epoch + 1}/{epochs}, Training accuracy:{training_accuracy}")
        print(f"Epoch {epoch + 1}/{epochs}: Total Loss {total_loss:.4f}")
        write.add_scalar("Loss_function", total_loss, epoch)

        # save model parameter
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), best_model_path)

        # evaluate in the test dataset
        if epoch % 2 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            total_test_accuracy = 0
            with torch.no_grad():
                for (Gain, Phase, label) in test_dataset_loader:
                    Gain, Phase, label = Gain.to(device), Phase.to(device), label.to(device)
                    output = model(Gain, Phase)
                    accuracy = (output.argmax(1) == label).sum()
                    total_test_accuracy += accuracy
                    preds = output.argmax(1).cpu().numpy()
                    labels = label.cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)

                precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)


                print(f"Epoch:{epoch + 1}/{epochs}, Test accuracy:{total_test_accuracy / test_dataset.number_of_sample}")
                print(f"Epoch:{epoch + 1}/{epochs}, Test precision:{precision}")
                print(f"Epoch:{epoch + 1}/{epochs}, Test recall:{recall}")
                print(f"Epoch:{epoch + 1}/{epochs}, Test f1:{f1}")
                write.add_scalar("test_accuracy", total_test_accuracy / test_dataset.number_of_sample, epoch)
                write.add_scalar("test_precision", precision, epoch)
                write.add_scalar("test_recall", recall, epoch)
                write.add_scalar("test_f1", f1, epoch)

                # Confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image = PIL.Image.open(buf)
                image = torchvision.transforms.ToTensor()(image)
                write.add_image("confusion_matrix", image, epoch)
                plt.close(fig)


                if whether_use_noise_testdat_to_evaluate:
                    for i, noise_test_dataset_loader in enumerate([noise_test_dataset_loader_1, noise_test_dataset_loader_2, noise_test_dataset_loader_3], start=1):
                        all_preds = []
                        all_labels = []
                        total_test_accuracy = 0
                        for (Gain, Phase, label) in noise_test_dataset_loader:
                            Gain, Phase, label = Gain.to(device), Phase.to(device), label.to(device)
                            output = model(Gain, Phase)
                            accuracy = (output.argmax(1) == label).sum()
                            total_test_accuracy += accuracy
                            preds = output.argmax(1).cpu().numpy()
                            labels = label.cpu().numpy()
                            all_preds.extend(preds)
                            all_labels.extend(labels)

                        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                        print(f"Epoch:{epoch + 1}/{epochs}, Noise test dataset {i} accuracy:{total_test_accuracy / noise_test_dataset_loader.dataset.number_of_sample}")
                        print(f"Epoch:{epoch + 1}/{epochs}, Noise test dataset {i} precision:{precision}")
                        print(f"Epoch:{epoch + 1}/{epochs}, Noise test dataset {i} recall:{recall}")
                        print(f"Epoch:{epoch + 1}/{epochs}, Noise test dataset {i} f1:{f1}")
                        write.add_scalar(f"noise_test_dataset_{i}_accuracy", total_test_accuracy / noise_test_dataset_loader.dataset.number_of_sample, epoch)
                        write.add_scalar(f"noise_test_dataset_{i}_precision", precision, epoch)
                        write.add_scalar(f"noise_test_dataset_{i}_recall", recall, epoch)
                        write.add_scalar(f"noise_test_dataset_{i}_f1", f1, epoch)

        ## VAT visualization
        if epoch % 10 == 0:
            r_vadv = virtual_adversarial_perturbation(Gain, Phase, model)
            perturbed = Gain + r_vadv
            diff = (perturbed - Gain).abs()
            img_grid = torchvision.utils.make_grid(diff[:8].cpu(), nrow=4, normalize=True)
            write.add_image("VAT_diff", img_grid, epoch)


    return model



def Select_samples_based_on_small_loss_criterion(record_losses_file, loss_function_sheet_name="Loss_function",
                                                 whether_for_auxiliary_model=False, average_epoch=5, loss_threshold=0.3):
    """
    select samples based on small loss criterion
    record_losses_file: file
    average_epoch : number of epochs to average over
    loss_threshold: small loss criterion threshold
    """

    loss_data = pd.read_excel(record_losses_file, sheet_name=loss_function_sheet_name, header=0)
    loss_data = np.array(loss_data)
    mean_loss = np.mean(loss_data[:, -average_epoch:], axis=1)  # Avg loss over last average epochs (stable phase)
    simple_mask = mean_loss < loss_threshold
    simple_sample_idx = np.where(simple_mask)[0]  # Simple samples
    hard_and_noise_sample_idx = np.where(~simple_mask)[0]  # Hard/Noisy

    if whether_for_auxiliary_model==False:

        print("------------------Information about selected samples--------------------")
        print(f"Simple samples: {len(simple_sample_idx)} ({len(simple_sample_idx) / loss_data.shape[0]*100:.2f}%);")
        print(f"Hard/Noisy: {len(hard_and_noise_sample_idx)} ({len(hard_and_noise_sample_idx)/loss_data.shape[0]*100:.2f}%)")
        print("------------------------------------------------------------------------")

        # sava data into excel file
        simple_sample_idx = pd.DataFrame(simple_sample_idx, columns=["Sample_ID"])
        hard_and_noise_sample_idx = pd.DataFrame(hard_and_noise_sample_idx, columns=["Sample_ID"])
        with pd.ExcelWriter(record_losses_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            simple_sample_idx.to_excel(writer, sheet_name="Simple_sample_index", index=False)
            hard_and_noise_sample_idx.to_excel(writer, sheet_name="hard_and_noise_sample_index", index=False)
    else:
        data = pd.read_excel(record_losses_file, sheet_name="Dataset_information_for_AM", header=0)
        Artifical_dataset_label_type =  data['Artifical_dataset_label_type']
        noise_sample = Artifical_dataset_label_type == "noise"
        noise_sample_index = np.where(noise_sample)[0]
        duplicates = [x for x in noise_sample_index if x in hard_and_noise_sample_idx]
        hard_sample_index = [x for x in hard_and_noise_sample_idx if x not in duplicates]
        hard_sample_label = ["hard" for x in hard_sample_index]
        noise_sample_label = ["noise" for x in noise_sample_index]

        hard_data_temp = pd.DataFrame({
            "Auxiliary_model_data_index": hard_sample_index,
            "Auxiliary_model_label": hard_sample_label,
        })
        noise_data_temp = pd.DataFrame({
            "Auxiliary_model_data_index": noise_sample_index,
            "Auxiliary_model_label": noise_sample_label,
        })
        Auxiliary_model_dataset = pd.concat([hard_data_temp, noise_data_temp])
        Auxiliary_model_dataset = Auxiliary_model_dataset.sample(frac=1, random_state=None)
        with pd.ExcelWriter(record_losses_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            Auxiliary_model_dataset.to_excel(writer, sheet_name="Auxiliary_model_dataset", index=False)


def use_auxiliary_model_to_identify_hard_or_noisy_sample(au_model, model_parameter_file, record_losses_file, device, loss_sequence_size, model_type="SVM"):
    all_loss_data = pd.read_excel(record_losses_file, sheet_name="Loss_function", header=0)
    all_loss_data = np.array(all_loss_data)
    hard_and_noise_sample_index_pd = pd.read_excel(record_losses_file, sheet_name="hard_and_noise_sample_index", header=0)
    hard_and_noise_sample_index = hard_and_noise_sample_index_pd["Sample_ID"].tolist()
    hard_and_noise_loss_data = all_loss_data[hard_and_noise_sample_index, :]
    if model_type == "LSTM":
        AM = Auxiliary_model(loss_sequence_size=loss_sequence_size)
        AM.load_state_dict(torch.load(model_parameter_file))  # weights_only=False 以加载整个模型
        AM = AM.to(device)
        AM.eval()
        hard_and_noise_loss_data = torch.tensor(hard_and_noise_loss_data, dtype=torch.float32).to(device)
        output = AM(hard_and_noise_loss_data)
        hard_and_noise_predict = output.argmax(1).detach().cpu().numpy()
    else:
        hard_and_noise_predict = au_model.predict(hard_and_noise_loss_data)
    hard_and_noise_sample_label_pd = pd.DataFrame({"hard(1)_and_noise(0)_predict":hard_and_noise_predict})
    hard_and_noise_sample_pd = pd.concat([hard_and_noise_sample_index_pd, hard_and_noise_sample_label_pd], axis=1)

    with pd.ExcelWriter(record_losses_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        hard_and_noise_sample_pd.to_excel(writer, sheet_name='hard_and_noise_sample_index', index=False)


def main():

    # initial setting
    origin_training_set_address = "Training_dataset_3"
    test_set_file_name = "Test_dataset_3"
    noise_dataset_name = "Test_dataset_3"
    whether_create_dataset = False
    whether_use_noise_testdat_to_evaluate = True
    device = intital_setup_and_seed(3407)
    epochs = 15
    batch_size = 32
    learning_rate = 0.001


    # Step 1: construct dataset and train backbone model in the first time
    if whether_create_dataset:
        create_training_and_test_dataset(origin_training_set_address=origin_training_set_address,
                                         training_set_file_name=f"./data_temp_log/{origin_training_set_address}.xlsx",
                                         test_set_file_name=f"./data_temp_log/{test_set_file_name}.xlsx",
                                         test_set_rate=0.3, hard_rate=0.2, noisy_rate=0.1)

    training_dataset = SFRA_dataset(whether_is_training=True,
                        training_set_file_name=f"./data_temp_log/{origin_training_set_address}.xlsx",
                        test_set_file_name=f"./data_temp_log/{test_set_file_name}.xlsx",
                        resize=128)

    test_dataset = SFRA_dataset(whether_is_training=False,
                                training_set_file_name=f"./data_temp_log/{origin_training_set_address}.xlsx",
                                test_set_file_name=f"./data_temp_log/{test_set_file_name}.xlsx",
                                resize=128)


    train_backbone_model_with_dynamics(training_dataset=training_dataset,
                                       test_dataset=test_dataset,
                                       device=device,
                                       training_set_file_name=f"./data_temp_log/{origin_training_set_address}.xlsx",
                                       logs_dir=f"./logs/{origin_training_set_address}/step1", epochs=epochs,
                                       batch_size=batch_size, lr=learning_rate, record_losses=True,
                                       whether_use_noise_testdat_to_evaluate=True, noise_dataset_name=noise_dataset_name)

    # Select sample based on small loss criterion for a artifical_dataset
    Select_samples_based_on_small_loss_criterion(record_losses_file = f"./data_temp_log/{origin_training_set_address}.xlsx",
                                                 average_epoch=5, loss_threshold=0.5)
    # Build artifical_dataset

    use_selected_simple_samples_to_build_a_artifical_dataset(training_set_file_name=f"./data_temp_log/{origin_training_set_address}.xlsx",
                                                             noisy_rate=0.2)

    Artifical_dataset = Auxiliary_model_training_dataset_or_artifical_dataset(
                        training_set_file_name=f"./data_temp_log/{origin_training_set_address}.xlsx",
                        whether_is_artifical_dataset=True)


    # Step2: Retrain backbone model to obtain new loss function for auxiliary_model_training
    train_backbone_model_with_dynamics(training_dataset=Artifical_dataset,
                                       test_dataset=test_dataset,
                                       device=device,
                                       training_set_file_name=f"./data_temp_log/{origin_training_set_address}.xlsx",
                                       loss_function_sheet_name = "Loss_function_for_AM",
                                       logs_dir=f"./logs/{origin_training_set_address}/step2", epochs=epochs,
                                       batch_size=batch_size, lr=learning_rate, record_losses=True,
                                       whether_use_test_dataset=False, whether_use_noise_testdat_to_evaluate=False,
                                       noise_dataset_name=noise_dataset_name)

    Select_samples_based_on_small_loss_criterion(record_losses_file = f"./data_temp_log/{origin_training_set_address}.xlsx",
                                                 loss_function_sheet_name="Loss_function_for_AM", whether_for_auxiliary_model=True,
                                                 average_epoch=5,
                                                 loss_threshold=0.25)

    # Step 3 Train auxiliary model
    Auxiliary_model_dataset = Auxiliary_model_training_dataset_or_artifical_dataset(training_set_file_name=f"./data_temp_log/{origin_training_set_address}.xlsx",
                                                                                    whether_is_artifical_dataset=False)

    # Besides, we can use NN-based model as auxiliary model (LSTM is prone to overfit)
    # AM = train_auxiliary_model(Auxiliary_model_dataset, loss_sequence_size=15, device=device,
    #                       logs_dir=f"./logs/{origin_training_set_address}/step3",
    #                       epochs=100, batch_size=32, lr=0.001)

    aux_model = train_auxiliary_model_based_on_SVM_or_Randomforest(Auxiliary_model_dataset, model_type="SVM")
    use_auxiliary_model_to_identify_hard_or_noisy_sample(au_model=aux_model, model_parameter_file=f"./logs/{origin_training_set_address}/step3/best_aux_model.pth",
                                                         record_losses_file=f"./data_temp_log/{origin_training_set_address}.xlsx",
                                                         device=device, loss_sequence_size=epochs, model_type="SVM")

    # Step 4: VAT (semi-supervised learning)
    label_dataset = label_or_unlabel_dataset(whether_dataset_is_label=True, training_set_file_name=f"./data_temp_log/{origin_training_set_address}.xlsx")
    unlabel_dataset = label_or_unlabel_dataset(whether_dataset_is_label=False, training_set_file_name=f"./data_temp_log/{origin_training_set_address}.xlsx")
    Virtual_adversarial_training_for_backbone_model(label_dataset=label_dataset, unlabel_dataset=unlabel_dataset, test_dataset=test_dataset, device=device,
                                                    logs_dir=f"./logs/{origin_training_set_address}/step4", epochs=50, batch_size=batch_size, lr=learning_rate,
                                                    alpha=0.3, whether_use_noise_testdat_to_evaluate=whether_use_noise_testdat_to_evaluate,
                                                    noise_dataset_name=noise_dataset_name)

if __name__ == '__main__':
    main()