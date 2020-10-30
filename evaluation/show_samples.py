import numpy as np
import matplotlib.pyplot as plt


def show_samples(experiment_name, file_name, num=100):
    npz_path = rf'.\samples\{experiment_name}\{file_name}.npz'
    images = np.load(npz_path)["samples"][:num]

    plt.figure(figsize=(10, 10))
    for i in range(num):
        ax = plt.subplot(10, 10, i + 1)
        ax.axis('off')
        plt.imshow(images[i])
    plt.show()


def main():
    experiment_name = input("Please input experiment_name: ")
    file_name = input("Please input file_name: ")
    show_samples(experiment_name, file_name)
