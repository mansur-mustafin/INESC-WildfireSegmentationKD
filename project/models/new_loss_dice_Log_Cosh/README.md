Dice loss with Log_Cosh

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
        return torch.log(torch.cosh(dice_loss))
```

