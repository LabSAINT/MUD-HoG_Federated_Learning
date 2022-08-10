import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help='Default=64', type=int, default=64)
    parser.add_argument("--test_batch_size", help='Default=64',type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10,
        help='Default is 10 epochs.')
    parser.add_argument("--optimizer", type=str, default='SGD',
        help='Default is SGD optimizer')
    parser.add_argument("--lr", type=float, default=0.01,
        help="Learning rate of models. Default=0.01.")
    parser.add_argument("--momentum", type=float, default=0.5,
        help="Default momentum is 0.5.")
    parser.add_argument("--weight_decay", type=float, default=0.,
        help="Default weight_decay is 0.")
    parser.add_argument("--seed", type=int, default=1,
        help='Default is 1.')
    parser.add_argument("--log_interval", type=int, default=11,
        help='Default log_interval is 11.')
    parser.add_argument("-n", "--num_clients", type=int, default=10,
        help='Default is 10 clients.')
    parser.add_argument("--output_folder", type=str, default="experiments",
                        help="path to output folder, e.g. \"experiment\"")
    parser.add_argument("--dataset", type=str,
        choices=["mnist", "cifar", "cifar100", "imdb", "fashion_mnist"],
        default="mnist", help="Default is mnist dataset.")
    parser.add_argument("--loader_type", type=str,
        choices=["iid", "byLabel", "dirichlet"], default="iid",
        help="Default is iid data distribution.")
    parser.add_argument("--loader_path", type=str, default="./data/loader.pk",
        help="where to save the data partitions. Default is ./data/loader.pk")
    parser.add_argument("--AR", type=str, default='fedavg',
        choices=['fedavg', 'median', 'gm', 'krum', 'mkrum', 'foolsgold',
            'residualbase', 'attention', 'mlp', 'mudhog', 'fedavg_oracle'],
        help="Aggregation rule. Default is fedavg.")
    parser.add_argument("--n_attacker_backdoor", type=int, default=0,
        help="Default is no attacker backdoor.")
    parser.add_argument("--backdoor_trigger", nargs="*",  type=int,
        default=[0,0,1,1], help="the hyperparameter for backdoor trigger, do `--backdoor_trigger x_offset y_offset x_interval y_interval`")
    parser.add_argument("--n_attacker_semanticBackdoor", type=int, default=0)
    parser.add_argument("--n_attacker_labelFlipping", type=int, default=0)
    parser.add_argument("--n_attacker_labelFlippingDirectional", type=int, default=0)
    parser.add_argument("--n_attacker_omniscient", type=int, default=0)
    parser.add_argument("--omniscient_scale", type=int, default=1)
    parser.add_argument("--attacks", type=str,
        help="if contains \"backdoor\", activate the corresponding tests")
    parser.add_argument("--save_model_weights", action="store_true")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--path_to_aggNet", type=str)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default='cuda')
    parser.add_argument("--inner_epochs", type=int, default=1)
    # Mao add for MUD-HoG
    parser.add_argument("--verbose", action="store_true")
    #======================UNRELIABLE clients================================
    parser.add_argument("--list_unreliable", type=str,
        help="List of unreliable clients, separate by comma \',\', e.g., \"1,2\"."
            " Default is None.")
    parser.add_argument("--unreliable_fraction", type=float, default=0.5,
        help="Fraction of local dataset is added noise to simulate unreliable"
            " client. Default is 0.5.")
    parser.add_argument("--unreliable_fracTrain", type=float, default=0.3,
        help="Fraction of local dataset is used to trained the local model."
            " Default is 0.3.")
    parser.add_argument("--mean_unreliable", type=float, default=0.,
        help="Mean of Gaussian noise add to raw image for unreliable clients."
            " Default is 0")
    parser.add_argument("--blur_method", type=str, default='gaussian_smooth',
        choices=['add_noise', 'gaussian_smooth'],
        help="Method to make low quality image (use std=50). Default: gaussian_smooth")
    parser.add_argument("--max_std_unreliable", type=float, default=50,
        help="Max Standard deviation of Gaussian noise add to raw image (below 1),"
            " or generate weight Gaussian smooth filter (std=50)"
            " for unreliable clients."
            " Value in (0,1]. Default is 50 (for gaussian_smooth).")
    #======================= ADDITIVE NOSE attacker=========================
    parser.add_argument("--list_uatk_add_noise", type=str,
        help="List of untargeted attacks, sending: Gaussian noise + gradient updates.")
    parser.add_argument("--mean_add_noise", type=float, default=0.,
        help="Mean of additive Gaussian noise add to grad for untargeted attack. Default is 0.")
    parser.add_argument("--std_add_noise", type=float, default=0.01,
        help="Standard deviation of additive Gaussian noise add to grad for untargeted attack."
            " Default is 0.01.")
    #=====================SIGN-FLIPPING Gradients attacker====================
    parser.add_argument("--list_uatk_flip_sign", type=str,
        help="List of untargeted attacks, sending flipping sign of gradient updates.")

    #=====================LABEL-FLIPPING TARGETED attacker====================
    parser.add_argument("--list_tatk_label_flipping", type=str,
        help="List of targeted attacks, change label from source_label(1) to target_label(7).")
    parser.add_argument("--list_tatk_multi_label_flipping", type=str,
        help="List of targeted attacks, change few labels from source_label(1,2,3) to target_label(7).")
    parser.add_argument("--source_labels", type=str, default ="1",
        help="Default of source label flipping is 1.")
    parser.add_argument("--target_label", type=int, default =7,
        help="Default of source label flipping is 7.")
    parser.add_argument("--list_tatk_backdoor", type=str,
        help="List of targeted backdoor attack, manipulate label 2 in MNIST."
             " Same meaning as n_attacker_backdoor, but list input option.")

    args = parser.parse_args()

    n = args.num_clients

    m = args.n_attacker_backdoor
    args.attacker_list_backdoor = np.random.permutation(list(range(n)))[:m]

    m = args.n_attacker_semanticBackdoor
    args.attacker_list_semanticBackdoor = np.random.permutation(list(range(n)))[:m]

    m = args.n_attacker_labelFlipping
    args.attacker_list_labelFlipping = np.random.permutation(list(range(n)))[:m]

    m = args.n_attacker_labelFlippingDirectional
    args.attacker_list_labelFlippingDirectional = np.random.permutation(list(range(n)))[:m]

    m = args.n_attacker_omniscient
    args.attacker_list_omniscient = np.random.permutation(list(range(n)))[:m]

    m = args.list_unreliable
    args.list_unreliable = [int(i) for i in m.split(',')] if m else []
    m = args.list_uatk_add_noise
    args.list_uatk_add_noise = [int(i) for i in m.split(',')] if m else []
    m = args.list_uatk_flip_sign
    args.list_uatk_flip_sign = [int(i) for i in m.split(',')] if m else []
    m = args.list_tatk_multi_label_flipping
    args.list_tatk_multi_label_flipping = [int(i) for i in m.split(',')] if m else []
    m = args.list_tatk_label_flipping
    args.list_tatk_label_flipping = [int(i) for i in m.split(',')] if m else []
    m = args.list_tatk_backdoor
    args.list_tatk_backdoor = [int(i) for i in m.split(',')] if m else []
    m = args.source_labels
    args.source_labels = [int(i) for i in m.split(',')] if m else [1]

    if args.experiment_name == None:
        args.experiment_name = f"{args.loader_type}/{args.attacks}/{args.AR}"

    return args


if __name__ == "__main__":

    import _main

    args = parse_args()
    print("#" * 64)
    for i in vars(args):
        print(f"#{i:>40}: {str(getattr(args, i)):<20}#")
    print("#" * 64)
    _main.main(args)
