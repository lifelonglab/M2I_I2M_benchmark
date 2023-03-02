import torch
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training import SynapticIntelligence, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC, CoPE, LFL, \
    MAS, Naive, GenerativeReplay, ICaRL
from avalanche.training.utils import get_last_fc_layer
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Compose, ToTensor, Normalize

icarl_augment_data = Compose([
    ToTensor(),
    Normalize(mean=(0.9221,), std=(0.2681,))
])


def parse_strategy_name(args, model, optimizer, criterion, device, eval_plugin):
    if args.strategy_name == 'LwF':
        assert (
                len(args.lwf_alpha) == 1 or len(args.lwf_alpha) == 5
        ), "Alpha must be a non-empty list."
        return LwF(
            model,
            optimizer,
            criterion,
            alpha=args.lwf_alpha[0] if len(args.lwf_alpha) == 1 else args.lwf_alpha,
            temperature=args.softmax_temperature,
            train_epochs=args.epochs,
            device=device,
            train_mb_size=args.minibatch_size,
            evaluator=eval_plugin
        )
    elif args.strategy_name == 'EWC':
        return EWC(
            model,
            optimizer,
            criterion,
            ewc_lambda=args.ewc_lambda,
            train_epochs=args.epochs,
            device=device,
            mode='separate',
            decay_factor=None,
            train_mb_size=args.minibatch_size,
            evaluator=eval_plugin,
            eval_every=1
        )
    elif args.strategy_name == 'SI':
        return SynapticIntelligence(
            model,
            optimizer,
            criterion,
            si_lambda=args.si_lambda,
            eps=args.si_eps,
            train_epochs=args.epochs,
            device=device,
            train_mb_size=args.minibatch_size,
            evaluator=eval_plugin,
        )
    elif args.strategy_name == 'ICaRL':
        model: IcarlNet = make_icarl_net(num_classes=args.num_classes)
        model.apply(initialize_icarl_net)
        return ICaRL(
            model.feature_extractor,
            model.classifier,
            optimizer=optimizer,
            train_epochs=args.epochs,
            device=device,
            train_mb_size=args.minibatch_size,
            evaluator=eval_plugin,
            memory_size=args.icarl_mem_size,
            fixed_memory=True,
            eval_every=1,
            buffer_transform=icarl_augment_data,
        )
    elif args.strategy_name == 'GEM':
        return GEM(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            patterns_per_exp=args.patterns_per_exp,
            train_epochs=args.epochs,
            device=device,
            train_mb_size=args.minibatch_size,
            evaluator=eval_plugin,
            eval_every=1
        )
    elif args.strategy_name == 'GenerativeReplay':
        return GenerativeReplay(
            model,
            torch.optim.Adam(model.parameters(), lr=0.001),
            CrossEntropyLoss(),
            train_mb_size=100,
            train_epochs=4,
            eval_mb_size=100,
            device=device,
            evaluator=eval_plugin,
        )
    elif args.strategy_name == 'CWRStar':
        return CWRStar(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.minibatch_size,
            train_epochs=args.epochs,
            eval_every=1,
            evaluator=eval_plugin,
            device=device,
            cwr_layer_name=get_last_fc_layer(model)[0]
        )
    elif args.strategy_name == 'Replay':
        return Replay(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.minibatch_size,
            train_epochs=args.epochs,
            eval_every=1,
            evaluator=eval_plugin,
            device=device,
            mem_size=args.replay_mem_size
        )
    elif args.strategy_name == 'GDumb':
        return GDumb(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.minibatch_size,
            train_epochs=args.epochs,
            eval_every=1,
            evaluator=eval_plugin,
            device=device
        )
    elif args.strategy_name == 'Cumulative':
        return Cumulative(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.minibatch_size,
            train_epochs=args.epochs,
            eval_every=1,
            evaluator=eval_plugin,
            device=device
        )
    elif args.strategy_name == 'AGEM':
        return AGEM(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.minibatch_size,
            train_epochs=args.epochs,
            eval_every=1,
            evaluator=eval_plugin,
            device=device,
            patterns_per_exp=args.patterns_per_exp
        )
    elif args.strategy_name == 'CoPE':
        return CoPE(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.minibatch_size,
            train_epochs=args.epochs,
            eval_every=1,
            evaluator=eval_plugin,
            device=device
        )
    elif args.strategy_name == 'LFL':
        return LFL(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.minibatch_size,
            train_epochs=args.epochs,
            eval_every=1,
            evaluator=eval_plugin,
            device=device,
            lambda_e=args.lfl_lambda,
        )

    elif args.strategy_name == 'MAS':
        return MAS(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.minibatch_size,
            train_epochs=args.epochs,
            eval_every=1,
            evaluator=eval_plugin,
            device=device,
            lambda_reg=args.mas_lambda_reg,
        )
    elif args.strategy_name == 'Naive':
        return Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.minibatch_size,
            train_epochs=args.epochs,
            eval_every=1,
            evaluator=eval_plugin,
            device=device,
        )

    else:
        raise NotImplementedError("STRATEGY NOT IMPLEMENTED YET: ", args.strategy_name)
