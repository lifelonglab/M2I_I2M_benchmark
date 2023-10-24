from avalanche.models.generator import MlpVAE, VAE_loss
from avalanche.training import SynapticIntelligence, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC, CoPE, LFL, \
    MAS, Naive, GenerativeReplay, VAETraining, PNNStrategy
from avalanche.training.plugins import GenerativeReplayPlugin
from avalanche.training.utils import get_last_fc_layer


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
    elif args.strategy_name == 'PNN':
        return PNNStrategy(model,
                           optimizer,
                           criterion,
                           train_epochs=args.epochs,
                           device=device,
                           train_mb_size=args.minibatch_size,
                           evaluator=eval_plugin,
                           eval_every=1)
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
    elif args.strategy_name == 'GenerativeReplay':
        generator = MlpVAE((3, 64, 64), nhid=16, device=device)
        # optimzer:
        lr = 0.01
        from torch.optim import Adam

        optimizer_generator = Adam(
            filter(lambda p: p.requires_grad, generator.parameters()),
            lr=lr,
            weight_decay=0.0001,
        )
        # strategy (with plugin):
        generator_strategy = VAETraining(
            model=generator,
            optimizer=optimizer_generator,
            criterion=VAE_loss,
            train_mb_size=args.minibatch_size,
            train_epochs=16,
            eval_mb_size=args.minibatch_size,
            device=device,
            plugins=[
                GenerativeReplayPlugin(
                    replay_size=args.replay_mem_size,
                    increasing_replay_size=False,
                )
            ],
        )
        return GenerativeReplay(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.minibatch_size,
            train_epochs=args.epochs,
            eval_every=1,
            evaluator=eval_plugin,
            device=device,
            replay_size=args.replay_mem_size,
            generator_strategy=generator_strategy
        )

    else:
        raise NotImplementedError("STRATEGY NOT IMPLEMENTED YET: ", args.strategy_name)
