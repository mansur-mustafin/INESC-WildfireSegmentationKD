0.5 loss_KD + 0.5 loss_CEL

```py
for images, labels in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
    # for images, labels in self.train_dataloader:
    images = images.to(device)
    labels = labels.unsqueeze(1).float().to(device)

    with torch.no_grad():
        teacher_logit = self.teacher(images)['out']
    student_logit = self.student(images)['out']

    loss_CEL = self.criterion(student_logit, labels)


    s_prob = torch.sigmoid(student_logit / T)
    t_prob = torch.sigmoid(teacher_logit / T).detach()

    t_prob_bin = torch.cat([1 - t_prob, t_prob], dim=1)         # (N,2,H,W)
    s_prob_bin = torch.cat([
            torch.log(1 - s_prob + 10**-7),
            torch.log(s_prob     + 10**-7)
        ], dim=1)  

    N, _, H, W = teacher_logit.shape
    loss_KD = torch.sum(t_prob_bin * (t_prob_bin.log() - s_prob_bin)) / (N * H * W) * (T * T)

    # print(float(loss_KD.detach()))

    loss = alpha * loss_CEL + (1 - alpha) * loss_KD

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    running_loss += loss.item()
```