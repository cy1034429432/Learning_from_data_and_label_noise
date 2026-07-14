import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import pandas as pd
import os


def apply_morphological_operations(binary_mask, kernal_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernal_size, kernal_size))
    closed_mask = cv2.dilate(binary_mask, kernel)
    opened_mask = cv2.erode(closed_mask, kernel)
    return opened_mask

def plot_binary_mask_and_save_figure(compared_image_address, fault_type, whether_add_noise, noise_SNR_dB, baseline_image_address="HOHA200.csv", save_image_address="mask.png", number=0):

    def read_SFRA_csv_data(address):
        data = pd.read_csv(address, encoding='gb18030')
        data = np.array(data)
        frequency = np.array(data[1:502, 0], dtype=float).reshape(-1)
        Gain = np.array(data[1:502, 1], dtype=float).reshape(-1)
        Phase = np.array(data[1:502, 2], dtype=float).reshape(-1)
        return (frequency, Gain, Phase)

    def convert_binary_image_and_apply_morphological_operations(fault_type, origin_image_address, save_image_address="a.png"):
        img = cv2.imread(origin_image_address)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret2, mask = cv2.threshold(src=img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
        if fault_type!= "Normal":
            mask = apply_morphological_operations(mask)
        plt.clf()
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.savefig(save_image_address, bbox_inches='tight', pad_inches=0)



    (baseline_frequency, baseline_Gain, baseline_Phase) = read_SFRA_csv_data(baseline_image_address)
    if whether_add_noise:
        np.random.seed(42)
        # Gain noise
        signal_var = np.var(baseline_Gain)
        noise_std = np.sqrt(signal_var / (10 ** (noise_SNR_dB / 10)))
        noise = np.random.normal(0, noise_std, baseline_Gain.shape)
        baseline_Gain = baseline_Gain + noise
        baseline_Gain = np.maximum(baseline_Gain, -100)
        # Phase noise
        signal_var = np.var(baseline_Phase)
        noise_std = np.sqrt(signal_var / (10 ** (noise_SNR_dB / 10)))
        noise = np.random.normal(0, noise_std, baseline_Phase.shape)
        baseline_Phase = baseline_Phase + noise
        baseline_Phase = np.maximum(baseline_Phase, -100)


    (compared_frequency, compared_Gain, compared_Phase) = read_SFRA_csv_data(compared_image_address)

    plt.clf()
    matplotlib.pyplot.fill_between(baseline_frequency, baseline_Gain, compared_Gain, where=None, interpolate=True)
    plt.xlim([0, 1000])
    plt.xlabel("Frequency/kHz")
    plt.ylim([-50, 0])
    plt.ylabel("Gain/dB")
    plt.axis('off')
    plt.savefig("temp_Gain.png", bbox_inches='tight', pad_inches=0)
    convert_binary_image_and_apply_morphological_operations(fault_type, origin_image_address="temp_Gain.png", save_image_address=save_image_address+f"_Gain_{number}.png")

    plt.clf()
    matplotlib.pyplot.fill_between(baseline_frequency, baseline_Phase, compared_Phase, where=None, interpolate=True)
    plt.xlim([0, 1000])
    plt.xlabel("Frequency/kHz")
    plt.ylim([-180, 180])
    plt.ylabel("Phase/degree")
    plt.axis('off')
    plt.savefig("temp_Phase.png", bbox_inches='tight', pad_inches=0)
    convert_binary_image_and_apply_morphological_operations(fault_type, origin_image_address="temp_Phase.png", save_image_address=save_image_address+f"_Phase_{number}.png")


def main_creat_dataset():
    whether_add_noise = True
    noise_SNR_dB = 10
    file_path_temp_1 = "SFRA数据整理"
    fault_type_list = os.listdir(file_path_temp_1)
    for fault_type in fault_type_list:
        file_path_temp_2 = os.path.join(file_path_temp_1, fault_type)
        os.makedirs("Training_dataset/" + fault_type, exist_ok=True)
        single_file_name_list = os.listdir(file_path_temp_2)
        file_length = len(single_file_name_list)
        for i in range(file_length):
            compared_image_address = os.path.join(file_path_temp_2, single_file_name_list[i])
            plot_binary_mask_and_save_figure(compared_image_address, fault_type, whether_add_noise, noise_SNR_dB, baseline_image_address="HOHA200.csv", save_image_address="Training_dataset/" + fault_type + "/" + fault_type, number=i+1)


if __name__ == "__main__":
    main_creat_dataset()
