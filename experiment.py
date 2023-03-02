import argparse
import os
import pathlib
from datetime import datetime

from config.utils import load_config
from paths import ROOT_PATH, LOGS_PATH
from scenarios.run_scenario import run_scenario


def _configure_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lwf_alpha",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 1.333, 2.25, 3.2],
        help="Penalty hyperparameter for LwF. It can be either"
             "a list with multiple elements (one alpha per "
             "experience) or a list of one element (same alpha "
             "for all experiences).",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=float,
        default=1,
        help="Temperature for softmax used in distillation",
    )

    parser.add_argument("--model_name", type=str, default='wide_VGG9', help="model name")
    parser.add_argument("--num_classes", type=int, default=30, help="number of different classes")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
    parser.add_argument("--hs", type=int, default=256, help="MLP hidden size.")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=128, help="Minibatch size."
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Specify GPU id to use. Use CPU if -1.",
    )
    parser.add_argument(
        "--strategy_name",
        type=str,
        default='GEM',
        help="Specify the strategy which will be run.",
    )

    parser.add_argument(
        "--patterns_per_exp",
        type=str,
        default=256,
        help="Specify patterns_per_exp value.",
    )

    parser.add_argument(
        "--si_lambda",
        type=float,
        default=1,
        help="Specify si_lambda value.",
    )

    parser.add_argument(
        "--si_eps",
        type=float,
        default=0.001,
        help="Specify si_epsilon value.",
    )

    parser.add_argument(
        "--ewc_lambda",
        type=float,
        default=1.,
        help="Specify ewc_lambda value.",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default='mnist_to_imagenet',
        help="Specify scenario.",
    )
    parser.add_argument('--config',
                        help="configuration file *.yml",
                        type=str,
                        required=False, default='task_incremental/cumulative_config_param.yml')

    parser.add_argument(
        "--replay_mem_size",
        type=float,
        default=200.,
        help="Specify ewc_lambda value.",
    )

    parser.add_argument(
        "--num_class_from_imagenet",
        type=int,
        default=50,
        help="Specify ewc_lambda value.",
    )

    parser.add_argument(
        "--num_experiences",
        type=int,
        default=5,
        help="Specify ewc_lambda value.",
    )


    parser.add_argument(
        "--scenario_type",
        choices=['class_incremental', 'task_incremental'],
        default='task_incremental',
        help="is class incremental or task incremental",
    )

    parser.add_argument(
        "--resized",
        choices=['resized', 'original'],
        default='resized',
        help="should dataset be resized",
    )

    return parser


if __name__ == '__main__':
    parser = _configure_parser()

    args = parser.parse_args()
    opt = load_config(args.config)
    parser.set_defaults(**opt)
    args = parser.parse_args()

    root_path = os.path.abspath(ROOT_PATH)

    if args.strategy_name == 'Replay':
        out_dir_schema = f'ICCS/{args.scenario_type}/{args.scenario}_{args.balanced}/{args.strategy_name}/' \
                         f'{args.model_name}_epochs_{args.epochs}_lr_{args.lr}_momentum_{args.momentum}_replay_mem_size' \
                         f'_{args.replay_mem_size}_{datetime.now().strftime("%d:%m:%Y_%H:%M:%S")}'
    else:
        out_dir_schema = f'ICCS/{args.scenario_type}/{args.scenario}_{args.balanced}/{args.strategy_name}/' \
                         f'{args.model_name}_epochs_{args.epochs}_lr_{args.lr}_momentum_{args.momentum}_{datetime.now().strftime("%d:%m:%Y_%H:%M:%S")}'

    logs_dir = pathlib.Path(f'{os.path.abspath(LOGS_PATH)}/{out_dir_schema}')
    tb_logs_dir = pathlib.Path(f'{root_path}/tb_logs/{out_dir_schema}')

    logs_dir.mkdir(parents=True, exist_ok=True)
    tb_logs_dir.mkdir(parents=True, exist_ok=True)

    run_scenario(args, logs_dir, tb_logs_dir)
    print('====finish main====')
