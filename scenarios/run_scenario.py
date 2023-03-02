import json

import torch
import yaml
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics, timing_metrics, \
    bwt_metrics, class_accuracy_metrics
from avalanche.logging import TextLogger, CSVLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

from models.model_provider import parse_model_name
from scenarios.scenarios_providers import parse_scenario
from strategies.strategies_provider import parse_strategy_name


def run_scenario(args, log_dir, tb_log_dir):
    print(f'run strategy: {args.strategy_name}')
    print(f'run scenario : {args.scenario}' )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    print(f'Using model {args.model_name}')
    model = parse_model_name(args)


    _save_configs(args, model, log_dir)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()

    csv_logger, text_logger, tensorboard_logger = _get_loggers(args, log_dir, tb_log_dir)

    scenario = parse_scenario(args)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        class_accuracy_metrics(experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        timing_metrics(experience=True, stream=True),
        # forward_transfer_metrics(experience=True, stream=True),
        loggers=[text_logger, csv_logger, tensorboard_logger]
    )

    strategy = parse_strategy_name(args=args,
                                   model=model,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   device=device,
                                   eval_plugin=eval_plugin)

    print("Starting experiment...")
    results = []
    print('train')

    for i, train_batch_info in enumerate(scenario.train_stream):
        print(
            "Start training on experience ", train_batch_info.current_experience
        )
        print('Number of images in experience', len(train_batch_info.dataset))

        strategy.train(train_batch_info, num_workers=0)
        print(
            "End training on experience ", train_batch_info.current_experience
        )
        print("Computing accuracy on the test set")
        results.append(strategy.eval(scenario.test_stream[:]))


    all_metrics = eval_plugin.get_all_metrics()
    with open(f'{log_dir}/all_metrics.json', 'w') as f:
        json.dump(all_metrics, f)

def _save_configs(args, model, log_dir):
    with open(f'{log_dir}/config.yml', 'w') as f:
        yaml.dump(args, f)
    with open(f'{log_dir}/model.txt', 'w') as f:
        f.write(str(model))


def _get_loggers(args, log_dir, tb_log_dir):
    # interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open(f'{log_dir}/text_logs.txt', 'w'))
    csv_logger = CSVLogger(log_folder=log_dir)
    tensorboard_logger = TensorboardLogger(tb_log_dir=tb_log_dir)
    return csv_logger, text_logger, tensorboard_logger
