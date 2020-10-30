import torch
import numpy as np
from evaluation.evaluation_utils import get_config, load_generator, sample_labels
import datetime
import pyperclip


def generate_samples(config, generator, samples_num=10000, batch_size=50):
    """
    samples_num: 10000 is fine
    batch_size: shouldn't be too big to avoid OOM
    """
    target_dir = f"./samples/{config['experiment_name']}/"
    with torch.no_grad():
        noise_z = torch.FloatTensor(batch_size, config["dim_z"])
        new_noise = lambda: noise_z.normal_().cuda()
        yy = sample_labels(config["n_classes"], batch_size)
        generated_samples = torch.cat(
            [generator(new_noise(), yy).detach().cpu() for _ in range(int(samples_num / batch_size))]).numpy()
        generated_samples = np.moveaxis(generated_samples, 1, -1)
        generated_samples = (generated_samples + 1) / 2 * 255
        generated_samples = generated_samples.astype(np.uint8)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        pyperclip.copy(timestamp)
        print(timestamp, " (copied to your clipboard)")
        np.savez(target_dir + timestamp, samples=generated_samples)
        return timestamp


def main():
    cfg = get_config()
    g = load_generator(cfg)
    generate_samples(cfg, g)


if __name__ == '__main__':
    main()
