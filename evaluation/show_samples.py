import numpy as np
import matplotlib.pyplot as plt

experiment_name = "BigGAN_C10_seed0_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema"
file_name = "2020-10-22_20_47_00"
npz_path = rf'.\samples\{experiment_name}\{file_name}.npz'
num = 100
images = np.load(npz_path)["samples"][:num]
# images = np.moveaxis(images, 1, -1)
# images = (images + 1) / 2

plt.figure(figsize=(10, 10))
for i in range(num):
    ax = plt.subplot(10, 10, i + 1)
    ax.axis('off')
    plt.imshow(images[i])
plt.show()