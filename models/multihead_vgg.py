"""
See: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""
import torch
import torch.nn as nn
import torchvision
#############################
# Static params: Config
#############################
from avalanche.models import MultiHeadClassifier, MultiTaskModule

from models.VGGSlim import make_layers, cfg


class VGGSlimNoOutputLayer(torchvision.models.VGG):
    """
    Creates VGG feature extractor from config and custom classifier.
    """

    def __init__(self, config='11Slim', num_classes=50, init_weights=True,
                 classifier_inputdim=512 * 7 * 7, classifier_dim1=512, classifier_dim2=512, batch_norm=False,
                 dropout=False, img_input_channels=3):
        features = make_layers(cfg[config], batch_norm=batch_norm, img_input_channels=img_input_channels)
        super(VGGSlimNoOutputLayer, self).__init__(features)

        if hasattr(self, 'avgpool'):  # Compat Pytorch>1.0.0
            self.avgpool = torch.nn.Identity()

        if dropout:  # Same as in Pytorch default: print(models.vgg11_bn())
            self.classifier = nn.Sequential(
                nn.Linear(classifier_inputdim, classifier_dim1),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(classifier_dim1, classifier_dim2),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_inputdim, classifier_dim1),
                nn.ReLU(True),
                nn.Linear(classifier_dim1, classifier_dim2),
            )
        if init_weights:
            self._initialize_weights()


class MultiHeadVGG(MultiTaskModule):
    def __init__(self, config='11Slim', init_weights=True,
                 classifier_inputdim=512 * 7 * 7, classifier_dim1=512, classifier_dim2=512, batch_norm=False,
                 dropout=False, img_input_channels=3):
        super().__init__()
        self.vgg = VGGSlimNoOutputLayer(config=config, init_weights=init_weights,
                                        classifier_inputdim=classifier_inputdim, classifier_dim1=classifier_dim1,
                                        classifier_dim2=classifier_dim2, batch_norm=batch_norm, dropout=dropout, img_input_channels=img_input_channels)
        self.classifier = MultiHeadClassifier(classifier_dim2)

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        x = self.vgg(x)
        return self.classifier(x, task_label)
