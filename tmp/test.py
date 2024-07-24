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

def modify_lraspp_model(model, num_input_channels):
    '''
    Funtion to modify imput layers
    '''
    original_conv = model.backbone['0'][0]
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
    model.backbone['0'] = new_conv
    return model

@hydra.main(config_path="", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Configuration: ")
    print(OmegaConf.to_yaml(cfg))


    train_data = Sen2FireDataset(cfg.path.scene1)
    train_data.add_data_dir(cfg.path.scene2)
    val_dataset = Sen2FireDataset(cfg.path.scene3)

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    lraspp = lraspp_mobilenet_v3_large(weights=LRASPP_Weights.DEFAULT)
    lraspp = modify_lraspp_model(lraspp, num_input_channels=13)
    num_classes=2

    lraspp.classifier.low_classifier = nn.Conv2d(
        40, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )
    lraspp.classifier.high_classifier = nn.Conv2d(
        128, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lraspp = lraspp.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lraspp.parameters(), lr=0.0001)
    
    num_epochs = 50

    # Training loop
    for epoch in range(num_epochs):
        lraspp.train()
        running_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = lraspp(images)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Loss: {epoch_loss}')
        
        # Validation
        lraspp.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_predictions = []

        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = lraspp(images)['out']
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Collect predictions and true labels for metrics
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_val_labels.extend(labels.cpu().numpy().flatten())
                all_val_predictions.extend(preds.flatten())
        
        val_loss /= len(val_dataloader)
        print(f'Validation Loss: {val_loss}')

        # Calculate and print metrics for the validation set
        if epoch + 1 == num_epochs:
            val_f1 = f1_score(all_val_labels, all_val_predictions)
            print(f'Validation F1 Score: {val_f1}')


if __name__ == "__main__":
    main()

