# import wandb
import time
import json
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from landset8_loader import Landset8Dataset
from model_manager import ModelManager
from utils import *

# Login WANDD
# with open("wandb_secret.json", "r") as f:
#     secrets = json.load(f)
# wandb.login(key=secrets["WANDB_API_KEY"])


class FirePredict():

    def __init__(self, cfg: DictConfig, path=None) -> None:
        self.cfg = cfg

        self._create_data_loaders()

        # Teacher
        self.teacher = ModelManager(cfg, cfg.teacher.name.lower()).create_model()
        self.teacher.load_state_dict(torch.load(cfg.path.teacher_model))
        self.teacher.eval()

        # Student
        self.student = ModelManager(cfg, cfg.model.name.lower()).create_model()
        if path is not None:
            print(path, end="\t")
            self.student.load_state_dict(torch.load(path))

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.student.parameters(),
            lr=self.cfg.const.lr,
            weight_decay=self.cfg.const.weight_decay
        )

    def _create_data_loaders(self):
        train_dataset = Landset8Dataset(
            images_csv="./split/images_train.csv",
            dataset_dir=self.cfg.path.dataset,
            mask=self.cfg.const.mask,
            augment=False
        )

        val_dataset = Landset8Dataset(
            images_csv="./split/images_val.csv",
            dataset_dir=self.cfg.path.dataset,
            mask=self.cfg.const.mask,
            augment=False
        )

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.cfg.const.batch_size, shuffle=True, num_workers=8, drop_last=True
        )

        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.cfg.const.batch_size, shuffle=False, num_workers=8, drop_last=True
        )

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher = self.teacher.to(device)
        self.student = self.student.to(device)

        best_val_loss = float('inf')
        epsilon = 1e-14
        num_epochs = self.cfg.const.num_epochs
        T = self.cfg.distill.T
        alpha = self.cfg.distill.alpha

        # wandb.watch(self.student, log="all", log_freq=100)

        for epoch in range(num_epochs):
            self.student.train()
            running_loss = 0.0

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

            epoch_loss = running_loss / len(self.train_dataloader)

            print(f'Epoch {epoch+1}/{num_epochs} Training loss: {epoch_loss:.7f}')

            # wandb.log({"train_loss": epoch_loss})

            # Validation
            self.student.eval()
            val_loss = 0.0

            TP_all = np.zeros((2, 1))
            FP_all = np.zeros((2, 1))
            TN_all = np.zeros((2, 1))
            FN_all = np.zeros((2, 1))
            n_valid_sample_all = 0

            with torch.no_grad():
                for images, labels in self.val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.student(images)['out']

                    # Metrics
                    if self.cfg.const.num_classes == 1:
                        preds = (outputs[:, 0, :, :] > self.cfg.const.th_fire).cpu().numpy().astype(np.uint8).flatten()
                    else:
                        preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()

                    labels_np = labels.cpu().numpy().flatten()

                    TP, FP, TN, FN, n_valid_sample = eval_image(preds, labels_np, 2)
                    TP_all += TP
                    FP_all += FP
                    TN_all += TN
                    FN_all += FN
                    n_valid_sample_all += n_valid_sample

                    # Loss
                    labels = labels.unsqueeze(1).float()
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(self.val_dataloader)

            accuracy = (TP_all[1] + TN_all[1]) / (TP_all[1] + TN_all[1] + FP_all[1] + FN_all[1] + epsilon)
            precision = TP_all[1] / (TP_all[1] + FP_all[1] + epsilon)
            recall = TP_all[1] / (TP_all[1] + FN_all[1] + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)
            iou = TP_all[1] / (TP_all[1] + FP_all[1] + FN_all[1] + epsilon)

            print(f'Validation Loss: {val_loss:.5f}, Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1 Score: {f1[0]:.3f}, IoU: {iou[0]:.3f}, Acc: {accuracy[0]:.3f}')

            # wandb.log({
            #     "val_loss": val_loss,
            #     "val_acc": accuracy[0],
            #     "val_precision": precision[0],
            #     "val_recall": recall[0],
            #     "val_f1": f1[0]
            # })

            save_model = False

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model = True

            if save_model:
                path = self.cfg.path.output_model + f"{self.cfg.model.name}_{epoch}_val_{val_loss:.2f}_f1_{f1[0]:.2f}.pt"
                torch.save(self.student.state_dict(), path)
                print('Saved model to: ' + path)

    def test(self):
        test_dataset = Landset8Dataset(
            images_csv="./split/images_test.csv",
            dataset_dir=self.cfg.path.dataset,
            mask=self.cfg.const.mask,
            augment=False
        )
        test_dataloader = DataLoader(test_dataset, batch_size=self.cfg.const.batch_size, shuffle=False, num_workers=8, drop_last=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.student = self.student.to(device)
        self.student.eval()

        TP_all = np.zeros((2, 1))
        FP_all = np.zeros((2, 1))
        TN_all = np.zeros((2, 1))
        FN_all = np.zeros((2, 1))
        epsilon = 1e-14

        with torch.no_grad():
            # for images, labels in tqdm(test_dataloader):
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.student(images)['out']

                if self.cfg.const.num_classes == 1:
                    preds = (outputs[:, 0, :, :] > self.cfg.const.th_fire).cpu().numpy().astype(np.uint8).flatten()
                else:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
                labels_np = labels.cpu().numpy().flatten()

                TP, FP, TN, FN, _ = eval_image(preds, labels_np, 2)
                TP_all += TP
                FP_all += FP
                TN_all += TN
                FN_all += FN

        accuracy = (TP_all[1] + TN_all[1]) / (TP_all[1] + TN_all[1] + FP_all[1] + FN_all[1] + epsilon)
        precision = TP_all[1] / (TP_all[1] + FP_all[1] + epsilon)
        recall = TP_all[1] / (TP_all[1] + FN_all[1] + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = TP_all[1] / (TP_all[1] + FP_all[1] + FN_all[1] + epsilon)

        print(f' - Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1 Score: {f1[0]:.3f}, IoU: {iou[0]:.3f}, Accuracy: {accuracy[0]:.3f}')


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    # wandb.init(project="FirePrediction", config=OmegaConf.to_container(cfg, resolve=True), name=f"Run_{time.strftime('%Y%m%d-%H%M%S')}")
    print("Configuration: ")
    print(OmegaConf.to_yaml(cfg))

    # TRAINING
    # fp = FirePredict(cfg)
    # fp.train()
    # wandb.finish()

    print("\nTesting...\n")

    # TESTING
    for weight_path in get_model_paths(cfg.path.weights_test, cfg.model.name):
        fp = FirePredict(cfg, weight_path)
        fp.test()


if __name__ == "__main__":
    init_seeds()
    main()
