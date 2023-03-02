import torch
from avalanche.models import MultiTaskModule, MultiHeadClassifier
from efficientnet_pytorch import EfficientNet


class MultiHeadEfficientnetNet(MultiTaskModule):
    def __init__(self, config='efficientnet-b1', num_classes=250, img_input_channels=1):
        super().__init__()
        self.model = EfficientNet.from_pretrained(config, in_channels=img_input_channels, num_classes=num_classes)
        self.classifier = MultiHeadClassifier(num_classes)

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        x = self.model(x)
        return self.classifier(x, task_label)


class MultiHeadEfficientnetNetNotPretrained(MultiTaskModule):
    def __init__(self, config='efficientnet-b1', num_classes=250, img_input_channels=1):
        super().__init__()
        self.model = EfficientNet.from_name(config, in_channels=img_input_channels, num_classes=num_classes)
        self.classifier = MultiHeadClassifier(num_classes)

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        x = self.model(x)
        return self.classifier(x, task_label)
