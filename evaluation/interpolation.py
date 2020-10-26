import torchvision
import torch
import BigGAN as model
import datetime
import os
from utils import interp
from evaluation.generate_samples import sample_labels, get_config


def load_generator(config, name_suffix='_ema_best0', device='cuda'):
    generator = model.Generator(**config).to(device)
    generator.load_state_dict(torch.load(f"./weights/{config['experiment_name']}/G{name_suffix}.pth"), strict=True)
    return generator


def generate_interpolations(config, generator, fix_z=False, fix_y=False, num_per_sheet=16, num_midpoints=8,
                            device='cuda'):
    """

    Args:
        config: 配置
        generator: 生成器模型
        fix_z: 插值的两侧的样本的 z 相同
        fix_y: 插值的两侧的样本的类别相同
        num_per_sheet: 有几个插值
        num_midpoints: 每个插值有多少中间点
        device:

    Returns:

    """
    target_dir = f"./samples/{config['experiment_name']}/_interpolation/"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if fix_z:
        zs = torch.randn(num_per_sheet, 1, config["dim_z"], device=device)
        zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, config["dim_z"])
    else:
        zs = interp(torch.randn(num_per_sheet, 1, config["dim_z"], device=device),
                    torch.randn(num_per_sheet, 1, config["dim_z"], device=device),
                    num_midpoints).view(-1, config["dim_z"])
    if fix_y:
        ys = sample_labels(config["n_classes"], num_per_sheet).view(num_per_sheet, 1, -1)
        ys = ys.repeat(1, num_midpoints + 2, 1).view(-1, 1)
    else:
        ys = interp(sample_labels(config["n_classes"], num_per_sheet).view(num_per_sheet, 1, -1),
                    sample_labels(config["n_classes"], num_per_sheet).view(num_per_sheet, 1, -1),
                    num_midpoints).view(-1, 1)
    with torch.no_grad():
        generated_samples = generator(zs, ys).data.cpu()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        fix_info = '' + ('N' if not fix_z and not fix_y else '') + ('Z' if fix_z else '') + (
            'Y' if fix_y else '') + ' fixed'
        image_filename = f"{target_dir}/{fix_info}-{timestamp}.jpg"
        torchvision.utils.save_image(generated_samples, image_filename, nrow=num_midpoints + 2, normalize=True)


def main():
    cfg = get_config()
    print("Loading model...")
    g = load_generator(cfg)
    print("Loading done.")
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        generate_interpolations(cfg, g, fix_z, fix_y)


if __name__ == '__main__':
    main()
