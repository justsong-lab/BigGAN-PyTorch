from evaluation.evaluation_utils import get_config, load_generator, print_banner, get_logger, load_cifar_samples, \
    load_generated_samples, load_f100_samples
from evaluation.generate_samples import generate_samples
from evaluation.interpolation import generate_interpolations
from evaluation.show_samples import show_samples
from evaluation.inception_score import get_inception_score
from evaluation.fid import get_fid


def calculate_IS_and_FID(dataset='C10'):
    experiment_name = input("Please input experiment_name: ")
    file_name = input("Please input file_name: ")
    calculate_IS_and_FID(experiment_name, file_name, "Face100")
    print("loading generated_samples...")
    generated_samples = load_generated_samples(experiment_name, file_name)
    print("loading done")
    print("loading cifar samples...")
    if dataset == 'C10':
        real_samples, _ = load_cifar_samples(mode="train")
    elif dataset == 'Face100':
        real_samples, _ = load_f100_samples(mode='train')
    else:
        print("unsupported dataset")
        return
    print("loading done")
    # test_samples = load_cifar_samples(mode="test")
    generated_samples_IS = get_inception_score(generated_samples)
    baseline_IS = get_inception_score(real_samples)
    samples_FID = get_fid(real_samples, generated_samples)
    print(generated_samples_IS, baseline_IS, samples_FID)


def main():
    cfg = get_config()
    log_path = f"./logs/{cfg['experiment_name']}/"
    logger = get_logger(log_path)
    logger("**************** New Start ****************")
    logger(cfg)
    g = load_generator(cfg)
    print_banner("Generating samples")
    filename = generate_samples(cfg, g)
    logger(f"Samples generated: {filename}.npz")
    print_banner("Generating interpolations")
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        generate_interpolations(cfg, g, fix_z, fix_y)
    logger("Interpolations generated")
    del g
    # print_banner("Showing some generated samples")
    # show_samples(cfg["experiment_name"], filename)
    print_banner("Calculating IS")
    generated_samples = load_generated_samples(cfg["experiment_name"], filename)
    generated_samples_IS = get_inception_score(generated_samples)
    logger(f"Inception Score: {generated_samples_IS[0]:.6f}+{generated_samples_IS[1]:.6f}")
    print_banner("Calculating FID (training & fake samples)")
    real_samples, _ = load_f100_samples(mode='train')
    fid = get_fid(real_samples, generated_samples)
    logger(f"FID: {fid:.6f}")
    logger("**************** Done ****************")


if __name__ == '__main__':
    main()
