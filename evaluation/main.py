from evaluation.evaluation_utils import get_config, load_generator, print_banner
from evaluation.generate_samples import generate_samples
from evaluation.interpolation import generate_interpolations
from evaluation.show_samples import show_samples


def main():
    cfg = get_config()
    g = load_generator(cfg)
    print_banner("Generating samples")
    filename = generate_samples(cfg, g)
    print_banner("Generating interpolations")
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        generate_interpolations(cfg, g, fix_z, fix_y)
    print_banner("Showing some generated samples")
    show_samples(cfg["experiment_name"], filename)


if __name__ == '__main__':
    main()
