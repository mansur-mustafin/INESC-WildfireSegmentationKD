```py
class DiceLoss(nn.Module):

    def __init__(self, smooth = 1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.gama = gama

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

class FirePredict():

    def __init__(self, cfg: DictConfig, path = None) -> None:
        self.cfg = cfg
        self._create_data_loaders()
        self._setup_model()
        if path is not None:
            self._load_weights(path)
        self.criterion = CombinedLoss(weight_dice = 0.7, weight_ce = 0.3)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.cfg.const.lr,
            weight_decay=self.cfg.const.weight_decay)
```