import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    fcn_resnet50,
    fcn_resnet101,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large
)

from models.UNet import UNet
from models.ENet import ENet
from models.FDDWNet import FDDWNet
from models.CGNet import CGNet

class ModelManager:
    
    def __init__(self, cfg, model_name = ""):
        self.cfg = cfg

        if len(model_name) == 0:
            self.model_name = cfg.model.name.lower()
        else:
            self.model_name = model_name

        self.model_mappings = {
            "lraspp": (lraspp_mobilenet_v3_large, None, self._modify_lraspp_model),
            "fcn_resnet50": (fcn_resnet50, None, self._modify_fcn_model),
            "fcn_resnet101": (fcn_resnet101, None, self._modify_fcn_model),
            "deeplabv3_resnet50": (deeplabv3_resnet50, None, self._modify_deeplab_model),
            "deeplabv3_resnet101": (deeplabv3_resnet101, None, self._modify_deeplab_model),
            "deeplabv3_mobilenet_v3_large": (deeplabv3_mobilenet_v3_large, None, self._modify_deeplab_mobilenet_model),
            "unet": (self._no_op, None, self._no_op),
            "enet": (self._no_op, None, self._no_op),
            "fddwnet": (self._no_op, None, self._no_op),
            "cgnet": (self._no_op, None, self._no_op),
        }

    def create_model(self):
        model_name = self.model_name
        if model_name not in self.model_mappings:
            raise ValueError(f"Model '{model_name}' not found.")

        model_fn, weights, modify_fn = self.model_mappings[model_name]

        model = None

        if model_name == "unet":
            model = UNet(
                n_classes=self.cfg.const.num_classes,
                n_filters=self.cfg.const.n_filters,
                dropout=self.cfg.const.dropout,
                batchnorm=True,
                n_channels=self.cfg.const.num_input_channels
            )
        elif model_name == "enet":
            model = ENet(
                num_classes=self.cfg.const.num_classes,
                encoder_relu=False,
                decoder_relu=True
            )
        elif model_name == "fddwnet":
            model = FDDWNet(
                classes=self.cfg.const.num_classes
            )
        elif model_name == "cgnet":
            model = CGNet(
                classes=self.cfg.const.num_classes
            )
        else:
            model = model_fn(weights=weights)

            modify_fn(
                model,
                num_input_channels=self.cfg.const.num_input_channels,
                num_classes=self.cfg.const.num_classes
            )
        return model


    def load_weights(self, model, path):
        """
        Load model weights from `path`. 
        """
        try:
            model.load_state_dict(torch.load(path))
            print(path, end="\t")
        except FileNotFoundError:
            print(f"Path {path} does not exist. Skipping loading weights.")


    def _no_op(self, *args, **kwargs):
        pass


    def _modify_lraspp_model(self, model, num_input_channels, num_classes=2):
        '''
        Funtion to modify imput layers and classifier layers
        '''
        # Input
        original_layers = model.backbone['0']
        original_conv = original_layers[0]

        new_conv = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        # TODO: initial weights in this case?
        new_conv.weight.data[:, :original_conv.in_channels] = original_conv.weight.data.clone()

        new_bn = nn.BatchNorm2d(original_conv.out_channels, eps=0.001,
                                momentum=0.01, affine=True, track_running_stats=True)
        new_activation = nn.Hardswish()

        model.backbone['0'] = nn.Sequential(new_conv, new_bn, new_activation)

        # Classifier
        model.classifier.low_classifier = nn.Conv2d(
            40, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model.classifier.high_classifier = nn.Conv2d(
            128, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )


    def _modify_fcn_model(self, model, num_input_channels, num_classes=2):
        '''
        Funtion to modify imput layers and classifier layers
        '''
        # Input
        original_first_layer = model.backbone.conv1
        model.backbone.conv1 = nn.Conv2d(
            num_input_channels, 
            64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )

        model.backbone.conv1.weight.data[:, 1:4] = original_first_layer.weight.data.clone()
        model.backbone.conv1.weight.data[:, 0:0].zero_()
        model.backbone.conv1.weight.data[:, 5:].zero_()

        # Classifier
        model.classifier[4] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))


    def _modify_deeplab_model(self, model, num_input_channels, num_classes=2):
        '''
        Funtion to modify imput layers and classifier layers
        '''
        # Input
        original_first_layer = model.backbone.conv1
        model.backbone.conv1 = nn.Conv2d(
            num_input_channels, 
            64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )

        model.backbone.conv1.weight.data[:, 1:4] = original_first_layer.weight.data.clone()
        model.backbone.conv1.weight.data[:, 0:0].zero_()
        model.backbone.conv1.weight.data[:, 5:].zero_()

        # Classifier
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))


    def _modify_deeplab_mobilenet_model(self, model, num_input_channels, num_classes=2):
        '''
        Funtion to modify imput layers and classifier layers
        '''
        # Input
        original_layers = model.backbone['0']
        original_conv = original_layers[0]

        new_conv = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        new_conv.weight.data[:, :original_conv.in_channels] = original_conv.weight.data.clone()
        new_bn = nn.BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        new_activation = nn.Hardswish()

        model.backbone['0'] = nn.Sequential(new_conv, new_bn, new_activation)

        # Classifier
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
