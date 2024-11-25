L = (N/N1) * L1 + (N/N0) * L0

```py
class CustomLoss(nn.Module):
    
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.N_i = 512 * 512
        self.epsilon = 1

    def forward(self, outputs, labels):

        N = outputs.shape[0] # batch_size

        log_probs = F.log_softmax(outputs, dim=1)

        loss_per_pixel = -log_probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1) # Contains loss for each pixel

        counts_class1 = labels.view(N, -1).sum(dim=1).float()
        counts_class0 = self.N_i - counts_class1
        counts = torch.stack([counts_class0, counts_class1], dim=1) + self.epsilon

        weights_c = self.N_i / counts

        per_pixel_weights = weights_c.gather(1, labels.view(N, -1)) # weight for each pixel

        weighted_loss_per_pixel = loss_per_pixel.view(N, -1) * per_pixel_weights

        loss = weighted_loss_per_pixel.mean()

        return loss
```