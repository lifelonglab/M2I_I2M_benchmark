from efficientnet_pytorch import EfficientNet

from models.VGGSlim import VGGSlim, MultiHeadMLP
from models.multihead_efficentNet import MultiHeadEfficientnetNet, MultiHeadEfficientnetNetNotPretrained
from models.multihead_vgg import MultiHeadVGG


def parse_model_name(args):
    if args.model_name == 'wide_VGG9':
        return VGGSlim(config='wide_VGG9',
                       num_classes=args.num_classes,
                       batch_norm=True,
                       img_input_channels=3,
                       classifier_inputdim=512,
                       init_weights=False)
    elif args.model_name == 'base_VGG9':
        return VGGSlim(config='base_VGG9',
                       num_classes=args.num_classes,
                       batch_norm=True,
                       img_input_channels=3,
                       classifier_inputdim=4096,
                       init_weights=False)
    elif args.model_name == 'small_VGG9':
        return VGGSlim(config='small_VGG9',
                       num_classes=args.num_classes,
                       batch_norm=True,
                       img_input_channels=3,
                       classifier_inputdim=2048,
                       init_weights=False)
    elif args.model_name == 'wide_VGG99_multihead':
        return MultiHeadVGG(config='wide_VGG9',
                            batch_norm=True,
                            img_input_channels=3,
                            classifier_inputdim=512,
                            classifier_dim1=256,
                            classifier_dim2=128,
                            init_weights=False)
    elif args.model_name == 'EfficientNet':
        return EfficientNet.from_pretrained('efficientnet-b1', in_channels=3, num_classes=args.num_classes)
    elif args.model_name == 'EfficientNet_NotPretrained':
        return EfficientNet.from_name('efficientnet-b1', in_channels=3, num_classes=args.num_classes)
    elif args.model_name == 'EfficientNet_multihead':
        return MultiHeadEfficientnetNet(config='efficientnet-b1', num_classes=10, img_input_channels=3)
    elif args.model_name == 'EfficientNet_multihead_NotPretrained':
        return MultiHeadEfficientnetNetNotPretrained(config='efficientnet-b1', num_classes=10, img_input_channels=3)
    elif args.model_name == 'multi_head_mlp':
        return MultiHeadMLP(hidden_size=args.hs, hidden_layers=2)
    else:
        raise NotImplementedError("MODEL NOT IMPLEMENTED YET: ", args.model_name)
