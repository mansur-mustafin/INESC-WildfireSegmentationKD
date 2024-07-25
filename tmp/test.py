import os
import hydra
import torch
import torch.nn as nn
import numpy as np

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights as LRASPP_Weights
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from Sen2FireLoader import Sen2FireDataset


class FirePredict():

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._create_data_loaders()
        self._setup_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.lraspp.parameters(), lr=0.0001)


    def _setup_model(self):
        num_classes = 2

        self.lraspp = lraspp_mobilenet_v3_large(weights=LRASPP_Weights.DEFAULT)
        self._modify_lraspp_model(num_input_channels=13)
        self.lraspp.classifier.low_classifier = nn.Conv2d(
            40, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        self.lraspp.classifier.high_classifier = nn.Conv2d(
            128, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
    

    def _modify_lraspp_model(self, num_input_channels):
        '''
        Funtion to modify imput layers
        '''
        original_conv = self.lraspp.backbone['0'][0]
        new_conv = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        new_conv.weight.data[:, :original_conv.in_channels] = original_conv.weight.data
        if original_conv.bias is not None:
            new_conv.bias.data = original_conv.bias.data
        self.lraspp.backbone['0'] = new_conv


    def _create_data_loaders(self):
        train_data = Sen2FireDataset(self.cfg.path.scene1)
        train_data.add_data_dir(self.cfg.path.scene2)
        val_dataset = Sen2FireDataset(self.cfg.path.scene3)

        self.train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
        self.val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)


    def train(self, num_epochs = 20):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lraspp = self.lraspp.to(device)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.lraspp.train()
            running_loss = 0.0
            for images, labels in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.lraspp(images)['out']
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(self.train_dataloader)
            print(f'Loss: {epoch_loss}')
            
            # Validation
            self.lraspp.eval()
            val_loss = 0.0
            all_val_labels = []
            all_val_predictions = []

            with torch.no_grad():
                for images, labels in self.val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = self.lraspp(images)['out']
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    # Collect predictions and true labels for metrics
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_val_labels.extend(labels.cpu().numpy().flatten())
                    all_val_predictions.extend(preds.flatten())
            
            val_loss /= len(self.val_dataloader)
            print(f'Validation Loss: {val_loss}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_f1 = f1_score(all_val_labels, all_val_predictions) # TODO: check this 
                torch.save(self.lraspp.state_dict(), self.cfg.path.output_model + f"lraspp_{epoch}_val_{val_loss}_f1_{val_f1}.pt")
                print('Saved model to: ' + self.cfg.path.output_model + f"lraspp_{epoch}_val_{val_loss:.2f}_f1_{val_f1:.2f}")


@hydra.main(config_path="", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Configuration: ")
    print(OmegaConf.to_yaml(cfg))

    fp = FirePredict(cfg)
    fp.train()    


if __name__ == "__main__":
    main()

