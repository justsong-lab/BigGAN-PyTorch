import torch
import numpy as np
import BigGAN as model
import utils
import datetime
import torchvision


def sample_pseudo_labels_sparse(num_classes, batch_size):
    labels = np.random.randint(low=0, high=num_classes, size=batch_size)
    pseudo_labels = torch.from_numpy(labels)
    pseudo_labels = pseudo_labels.type(torch.long).cuda()
    return pseudo_labels


def main(config, device='cuda'):
    generator = model.Generator(**config).to(device)
    generator.load_state_dict(torch.load(f"./weights/{config['experiment_name']}/G.pth"), strict=True)
    target_dir = f"./samples/{config['experiment_name']}/"
    with torch.no_grad():
        samples_num = 10000
        batch_size = 250
        noise_z = torch.FloatTensor(batch_size, config["dim_z"])
        new_noise = lambda: noise_z.normal_().cuda()
        yy = sample_pseudo_labels_sparse(config["n_classes"], batch_size)
        generated_samples = torch.cat(
            [generator(new_noise(), yy).detach().cpu() for _ in range(int(samples_num / batch_size))]).numpy()
        generated_samples = np.moveaxis(generated_samples, 1, -1)
        generated_samples = (generated_samples + 1) / 2 * 255
        generated_samples = generated_samples.astype(int)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        print(timestamp)
        np.savez(target_dir + timestamp, samples=generated_samples)


if __name__ == '__main__':
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    main(config)
