import os
import hydra
import torch
import torch.nn as nn
import numpy as np

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, FCN_ResNet50_Weights, FCN_ResNet101_Weights, DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights, DeepLabV3_MobileNet_V3_Large_Weights
import torch.optim as optim
from tqdm import tqdm

from Sen2FireLoader import Sen2FireDataset
from utils import eval_image

class FirePredict():

    def __init__(self, cfg: DictConfig, path = None) -> None:
        self.cfg = cfg
        self._create_data_loaders()
        self._setup_model()
        if path is not None:
            self._load_weights(path)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.const.lr)


    def _setup_model(self):
        
        model_name = self.cfg.model.name.lower()

        if model_name == "lraspp":
            self.model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
            self._modify_lraspp_model(num_input_channels=13, num_classes=2)
        
        elif model_name == "fcn_resnet50":
            self.model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
            self._modify_fcn_model(num_input_channels=13, num_classes=2)

        elif model_name == "fcn_resnet101":
            self.model = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT)
            self._modify_fcn_model(num_input_channels=13, num_classes=2)

        elif model_name == "deeplabv3_resnet50":
            self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
            self._modify_deeplab_model(num_input_channels=13, num_classes=2)

        elif model_name == "deeplabv3_resnet101":
            self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
            self._modify_deeplab_model(num_input_channels=13, num_classes=2)

        elif model_name == "deeplabv3_mobilenet_v3_large":
            self.model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
            self._modify_deeplab_mobilenet_model(num_input_channels=13, num_classes=2)

        else:
            print(f"Model {model_name} was not found.")
            return
    

    def _load_weights(self, path):
        """
        Function to load model weights from a given path.
        """
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Model weights loaded from {path}")
        else:
            print(f"Path {path} does not exist. Skipping loading weights.")


    def _modify_lraspp_model(self, num_input_channels, num_classes = 2):
        '''
        Funtion to modify imput layers and classifier layers
        '''
        # Input
        original_layers = self.model.backbone['0']
        original_conv = original_layers[0]
        
        new_conv = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        new_conv.weight.data[:, :original_conv.in_channels] = original_conv.weight.data.clone() #TODO: initial weights in this case?

        new_bn = nn.BatchNorm2d(original_conv.out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        new_activation = nn.Hardswish()

        self.model.backbone['0'] = nn.Sequential(new_conv, new_bn, new_activation)

        # Classifire
        self.model.classifier.low_classifier = nn.Conv2d(
            40, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        self.model.classifier.high_classifier = nn.Conv2d(
            128, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )


    def _modify_fcn_model(self, num_input_channels, num_classes = 2):
        '''
        Funtion to modify imput layers and classifier layers
        '''
        # Input
        original_first_layer = self.model.backbone.conv1
        self.model.backbone.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.model.backbone.conv1.weight.data[:, 1:4] = original_first_layer.weight.data.clone()
        self.model.backbone.conv1.weight.data[:, 0:0].zero_()
        self.model.backbone.conv1.weight.data[:, 5:].zero_()

        # Classifire
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))


    def _modify_deeplab_model(self, num_input_channels, num_classes = 2):
        '''
        Funtion to modify imput layers and classifier layers
        '''
        # Input
        original_first_layer = self.model.backbone.conv1
        self.model.backbone.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.model.backbone.conv1.weight.data[:, 1:4] = original_first_layer.weight.data.clone()
        self.model.backbone.conv1.weight.data[:, 0:0].zero_()
        self.model.backbone.conv1.weight.data[:, 5:].zero_()

        # Classifire
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))


    def _modify_deeplab_mobilenet_model(self, num_input_channels, num_classes = 2):
        '''
        Funtion to modify imput layers and classifier layers
        '''
        # Input
        original_layers = self.model.backbone['0']
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

        self.model.backbone['0'] = nn.Sequential(new_conv, new_bn, new_activation)

        # Classifire
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))


    def _create_data_loaders(self):
        train_data = Sen2FireDataset(self.cfg.path.scene1)
        train_data.add_data_dir(self.cfg.path.scene2)
        val_dataset = Sen2FireDataset(self.cfg.path.scene3)

        self.train_dataloader = DataLoader(train_data, batch_size = self.cfg.const.batch_size, shuffle=True, num_workers=4)
        self.val_dataloader = DataLoader(val_dataset, batch_size = self.cfg.const.batch_size, shuffle=False, num_workers=4)


    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        best_val_loss = float('inf')
        epsilon = 1e-14
        num_epochs = self.cfg.const.num_epochs
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            # for images, labels in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            for images, labels in self.train_dataloader:
                images, labels = images.to(device), labels.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)['out']
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(self.train_dataloader)
            print(f'Epoch {epoch+1}/{num_epochs} Training loss: {epoch_loss:.4f}')
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            TP_all = np.zeros((2, 1))
            FP_all = np.zeros((2, 1))
            TN_all = np.zeros((2, 1))
            FN_all = np.zeros((2, 1))
            n_valid_sample_all = 0

            with torch.no_grad():
                for images, labels in self.val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = self.model(images)['out']
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
                    labels_np = labels.cpu().numpy().flatten()

                    TP, FP, TN, FN, n_valid_sample = eval_image(preds, labels_np, 2)
                    TP_all += TP
                    FP_all += FP
                    TN_all += TN
                    FN_all += FN
                    n_valid_sample_all += n_valid_sample
            
            val_loss /= len(self.val_dataloader)

            accuracy = (TP_all[1] + TN_all[1]) / (TP_all[1] + TN_all[1] + FP_all[1] + FN_all[1] + epsilon)
            precision = TP_all[1] / (TP_all[1] + FP_all[1] + epsilon)
            recall = TP_all[1] / (TP_all[1] + FN_all[1] + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)
            iou = TP_all[1] / (TP_all[1] + FP_all[1] + FN_all[1] + epsilon)

            print(f'Validation Loss: {val_loss:.3f}, Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1 Score: {f1[0]:.3f}, IoU: {iou[0]:.3f}, Acc: {accuracy[0]:.3f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                path = self.cfg.path.output_model + f"{self.cfg.model.name}_{epoch}_val_{val_loss:.2f}_f1_{f1[0]:.2f}.pt"
                torch.save(self.model.state_dict(), path)
                print('Saved model to: ' + path)


    def test(self):
        test_dataset = Sen2FireDataset(self.cfg.path.scene4)
        test_dataloader = DataLoader(test_dataset, batch_size=self.cfg.const.batch_size, shuffle=False, num_workers=4)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()

        TP_all = np.zeros((2, 1))
        FP_all = np.zeros((2, 1))
        TN_all = np.zeros((2, 1))
        FN_all = np.zeros((2, 1))
        epsilon = 1e-14

        with torch.no_grad():
            for images, labels in tqdm(test_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)['out']

                preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
                labels_np = labels.cpu().numpy().flatten()

                TP, FP, TN, FN, _ = eval_image(preds, labels_np, 2)
                TP_all += TP
                FP_all += FP
                TN_all += TN
                FN_all += FN

        accuracy = (TP_all[1] + TN_all[1]) / (TP_all[1] + TN_all[1] + FP_all[1] + FN_all[1] + epsilon)
        precision = TP_all[1] / (TP_all[1] + FP_all[1] + epsilon)
        recall = TP_all[1] / (TP_all[1] + FN_all[1] + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = TP_all[1] / (TP_all[1] + FP_all[1] + FN_all[1] + epsilon)

        print(f'Test Metrics - Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1 Score: {f1[0]:.3f}, IoU: {iou[0]:.3f}, Accuracy: {accuracy[0]:.3f}')


@hydra.main(config_path="", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Configuration: ")
    print(OmegaConf.to_yaml(cfg))

    fp = FirePredict(cfg, "models/1/lraspp_0_val_0.17_f1_0.00.pt")
    fp.test()
    fp = FirePredict(cfg, "models/1/lraspp_2_val_0.13_f1_0.19.pt")
    fp.test()
    fp = FirePredict(cfg, "models/1/lraspp_3_val_0.11_f1_0.39.pt")
    fp.test()
    fp = FirePredict(cfg, "models/1/lraspp_4_val_0.10_f1_0.41.pt")
    fp.test()

    # fp.train()
    
if __name__ == "__main__":
    main()
