# Dice Loss

* [1](https://dev.to/_aadidev/3-common-loss-functions-for-image-segmentation-545o)
* [2](https://arxiv.org/html/2312.05391v1#:~:text=2.2.,-1%20Dice%20Loss&text=The%20dice%20coefficient%20is%20commonly,the%20class%20distribution%20is%20imbalanced)

```py
class DiceLoss(nn.Module):

    def __init__(self, smooth = 1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.gama = gama

    def forward(self, outputs, labels):
        
        # print(outputs.shape)  torch.Size([8, 2, 512, 512])
        # print(labels.shape)   torch.Size([8, 512, 512])

        outputs = F.softmax(outputs, dim=1)             # The predictions are not taken as certain 0,1 but relaxed and taken as probabilities [0,1]. (From source 2)

        outputs = outputs.contiguous().view(-1, 2)      # 2D array
        labels = labels.view(-1)                        # 1D array

        labels_one_hot = F.one_hot(labels, num_classes=2)   # 2D array
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
```