import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

class DiceLoss(nn.Module):

    def __init__(self, smooth = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, labels):
        
        # print(outputs.shape)  torch.Size([8, 2, 512, 512])
        # print(labels.shape)   torch.Size([8, 512, 512])
        
        outputs = F.softmax(outputs, dim=1)

        outputs = outputs.contiguous().view(-1, 2)
        labels = labels.view(-1)

        labels_one_hot = F.one_hot(labels, num_classes=2)
        labels_one_hot = labels_one_hot.float()

        dice = 0
        for c in range(2):
            output_c = outputs[:, c]
            target_c = labels_one_hot[:, c]

            intersection = (output_c * target_c).sum()
            union = output_c.sum() + target_c.sum()

            dice_c = (2 * intersection + self.smooth) / (union + self.smooth)
            dice += dice_c

        dice_loss = 1 - dice / 2
        return dice_loss


class ExpLogLoss(nn.Module):
    
    def __init__(self, cfg: DictConfig, eps=1e-6):
        super(ExpLogLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.wce_loss = nn.CrossEntropyLoss(weight=cfg.get("class_weights", None))
        self.alpha = cfg.get("alpha", 0.5)  # Dice
        self.beta = cfg.get("beta", 0.5)    # WCE
        self.gamma_dice = cfg.get("gamma_dice", 1.5)
        self.gamma_wce = cfg.get("gamma_wce", 1.5)
        self.eps = eps

    def forward(self, outputs, labels):
        dice_loss = self.dice_loss(outputs, labels)
        wce_loss = self.wce_loss(outputs, labels)

        wce_loss = torch.exp(-wce_loss) # test

        dice_loss = torch.clamp(dice_loss, min=self.eps, max=1.0)
        wce_loss = torch.clamp(wce_loss, min=self.eps, max=1.0)

        exp_log_dice = (-torch.log(dice_loss)).pow(self.gamma_dice)
        exp_log_wce = (-torch.log(wce_loss)).pow(self.gamma_wce)

        loss = self.alpha * exp_log_dice + self.beta * exp_log_wce
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce


    def forward(self, outputs, targets):
        dice = self.dice_loss(outputs, targets)
        ce = self.ce_loss(outputs, targets)
        combined_loss = self.weight_dice * dice + self.weight_ce * ce

        print(f"loss: {combined_loss}, dice: {dice}, wce: {ce}")
        return combined_loss
