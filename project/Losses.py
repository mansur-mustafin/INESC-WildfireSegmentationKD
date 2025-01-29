import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
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
        self.wce_loss = nn.CrossEntropyLoss(
            weight=cfg.get("class_weights", None))
        self.alpha = cfg.get("alpha", 0.5)  # Dice
        self.beta = cfg.get("beta", 0.5)    # WCE
        self.gamma_dice = cfg.get("gamma_dice", 1.5)
        self.gamma_wce = cfg.get("gamma_wce", 1.5)
        self.eps = eps

    def forward(self, outputs, labels):
        dice_loss = self.dice_loss(outputs, labels)
        wce_loss = self.wce_loss(outputs, labels)

        wce_loss = torch.exp(-wce_loss)  # test

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

        return combined_loss


class FocalLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(FocalLoss, self).__init__()

        self.gamma = cfg.get("gamma_F", 1.5)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none')  # Cross-entropy
        pt = torch.exp(-ce_loss)  # Probabilities
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(FocalTverskyLoss, self).__init__()

        self.gamma = cfg.get("gamma_FT", 1.5)
        self.alpha = cfg.get("alpha_FT", 1.5)
        self.beta = cfg.get("beta_FT", 1.5)
        self.smooth = 1e-6

    def forward(self, input, target):
        input = F.softmax(input, dim=1)

        num_classes = input.size(1)
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target = target.view(-1)

        targets_one_hot = F.one_hot(target, num_classes=num_classes).float()

        losses = 0
        for c in range(num_classes):
            target_c = targets_one_hot[:, c]
            input_c = input[:, c]

            t_p = (input_c * target_c).sum()
            f_p = ((1 - target_c) * input_c).sum()
            f_n = (target_c * (1 - input_c)).sum()

            tversky = (t_p + self.smooth) / (t_p + self.alpha * f_p + self.beta * f_n + self.smooth)

            focal_tversky = (1 - tversky) ** self.gamma
            losses += focal_tversky

        loss = losses / num_classes
        return loss


class UnifiedFocalLoss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(UnifiedFocalLoss, self).__init__()

        self.focal_loss = FocalLoss(cfg)
        self.focal_tversky_loss = FocalTverskyLoss(cfg)
        self.weight_F = cfg.get("weight_F", 0.5)

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        focal_tversky_loss = self.focal_tversky_loss(inputs, targets)

        unified_loss = self.weight_F * focal_loss + (1 - self.weight_F) * focal_tversky_loss

        return unified_loss
