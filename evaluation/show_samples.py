import numpy as np
import matplotlib.pyplot as plt

experiment_name = "BigGAN_Face100_seed0_Gch64_Dch64_bs16_nDs2_Glr5.0e-05_Dlr2.0e-04_Gnlrelu_Dnlrelu_Ginitortho_Dinitortho_Gattn64_Dattn64_ema"
file_name = "2020-10-29_19_58_22"
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