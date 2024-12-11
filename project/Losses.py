import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth = 1e-6, gama=2, write_log = False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.gama = gama
        if write_log :
            print(f"""Used DiceLoss: 
    smooth = {smooth}
    gama = {gama}
            """)

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
    
    def __init__(self, alpha=0.5, beta=0.5, gamma_dice=1.5, gamma_wce=1.5, class_weights=None, eps=1e-6, write_log = False):
        super(ExpLogLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.alpha = alpha  # Dice
        self.beta = beta    # WCE
        self.gamma_dice = gamma_dice
        self.gamma_wce = gamma_wce
        self.class_weights = class_weights
        self.eps = eps
        if write_log :
            print(f"""Used DiceLoss: 
    alpha = {self.alpha}
    beta = {self.beta}
    gamma_dice = {self.gamma_dice}
    gamma_wce = {self.gamma_wce}
    class_weights = {self.class_weights}
            """)

    def forward(self, outputs, labels):
        dice_loss = self.dice_loss(outputs, labels)
        wce_loss = F.cross_entropy(outputs, labels, weight=self.class_weights, reduction='mean')

        dice_loss = torch.clamp(dice_loss, min=self.eps, max=1.0)
        wce_loss = torch.clamp(wce_loss, min=self.eps, max=1.0)

        exp_log_dice = (-torch.log(dice_loss)).pow(self.gamma_dice)
        exp_log_wce = (-torch.log(wce_loss)).pow(self.gamma_wce)

        loss = self.alpha * exp_log_dice + self.beta * exp_log_wce
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, write_log = False):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        if write_log :
            print(f"""Used CombinedLoss: 
    weight_dice = {weight_dice}
    weight_ce = {weight_ce}
            """)


    def forward(self, outputs, targets):
        dice = self.dice_loss(outputs, targets)
        ce = self.ce_loss(outputs, targets)
        combined_loss = self.weight_dice * dice + self.weight_ce * ce
        return combined_loss
