# --------------------------------------------------------------------------------
# 		Import
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
import torch.utils.model_zoo as model_zoo


# --------------------------------------------------------------------------------
#       Config
# --------------------------------------------------------------------------------
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
          512, 'M'],

    # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
    #       'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
          'M'],

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
          512, 512, 'M', 512, 512, 512, 512, 'M']
}


# --------------------------------------------------------------------------------
#       Funcs
# --------------------------------------------------------------------------------
def make_layers(cfg, batch_norm=False):
    """ Make layers for all VGG version from config
    """
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            # Add MaxPool2x layer if config have 'M'
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # Add batch norm
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                # Add ReLU
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
       Args:
       pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


# --------------------------------------------------------------------------------
#       Class
# --------------------------------------------------------------------------------
class VGG(nn.Module):
    """ VGG model to extract feature from image and it is driven by Transformer
        to extract sequence feature
    """
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 30))

        # # AdaptiveAvgPool will return fixed size for any input size
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # # Common fully connected layers
        # self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
        #                                 nn.ReLU(True),
        #                                 nn.Dropout(),
        #                                 nn.Linear(4096, 4096),
        #                                 nn.ReLU(True),
        #                                 nn.Dropout(),
        #                                 nn.Linear(4096, num_classes))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1, x.size(1))

        # x = self.avgpool(x)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
