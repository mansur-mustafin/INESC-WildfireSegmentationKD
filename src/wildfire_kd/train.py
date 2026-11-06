import time
from pathlib import Path

from dotenv import load_dotenv
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .datasets import Landset8Dataset
from .model_manager import ModelManager
from .utils import *

class FirePredict():

    def __init__(self, cfg: DictConfig, path=None, run=None) -> None:
        self.cfg = cfg
        self.run = run

        self._create_data_loaders()

        self.model_manager = ModelManager(cfg)
        self.model = self.model_manager.create_model()

        if path is not None:
            self.model_manager.load_weights(self.model, path)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.const.lr,
            weight_decay=self.cfg.const.weight_decay
        )

    def _create_data_loaders(self):
        train_dataset = Landset8Dataset(
            images_csv="./data/split/images_train.csv",
            dataset_dir=self.cfg.path.dataset,
            mask=self.cfg.const.mask,
            augment=False
        )

        val_dataset = Landset8Dataset(
            images_csv="./data/split/images_val.csv",
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.model = self.model.to(device)

        best_val_loss = float('inf')
        epsilon = 1e-14
        num_epochs = self.cfg.const.num_epochs

        if self.run is not None:
            self.run.watch(self.model, log="all", log_freq=100)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for images, labels in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # for images, labels in self.train_dataloader:
                images, labels = images.to(device), labels.to(device)

                labels = labels.unsqueeze(1).float()

                self.optimizer.zero_grad()
                outputs = self.model(images)['out']
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_dataloader)

            print(f'Epoch {epoch+1}/{num_epochs} Training loss: {epoch_loss:.4f}')

            if self.run is not None:
                self.run.log({"train_loss": epoch_loss})

            # Validation
            self.model.eval()
            val_loss = 0.0

            TP_all = np.zeros((2, 1))
            FP_all = np.zeros((2, 1))
            TN_all = np.zeros((2, 1))
            FN_all = np.zeros((2, 1))
            n_valid_sample_all = 0

            with torch.no_grad():
                for images, labels in self.val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)['out']

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

            if self.run is not None:
                self.run.log({
                    "val_loss": val_loss,
                    "val_acc": accuracy[0],
                    "val_precision": precision[0],
                    "val_recall": recall[0],
                    "val_f1": f1[0]
                })

            save_model = False

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model = True

            if save_model:
                path = self.cfg.path.output_model + f"{self.cfg.model.name}_{epoch}_val_{val_loss:.2f}_f1_{f1[0]:.2f}.pt"
                torch.save(self.model.state_dict(), path)
                print('Saved model to: ' + path)

    def test(self):
        test_dataset = Landset8Dataset(
            images_csv="./data/split/images_test.csv",
            dataset_dir=self.cfg.path.dataset,
            mask=self.cfg.const.mask,
            augment=False
        )
        test_dataloader = DataLoader(test_dataset, batch_size=self.cfg.const.batch_size, shuffle=False, num_workers=8, drop_last=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()

        TP_all = np.zeros((2, 1))
        FP_all = np.zeros((2, 1))
        TN_all = np.zeros((2, 1))
        FN_all = np.zeros((2, 1))
        epsilon = 1e-14

        with torch.no_grad():
            # for images, labels in tqdm(test_dataloader):
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)['out']

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


@hydra.main(config_path="../../conf", config_name="train", version_base=None)
def main(cfg: DictConfig):
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    run = None
    if cfg.wandb.mode != "disabled":
        import wandb
        run = wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"Run_{time.strftime('%Y%m%d-%H%M%S')}",
        )
    print("Configuration: ")
    print(OmegaConf.to_yaml(cfg))

    # TRAINING
    try:
        fp = FirePredict(cfg, run=run)
        fp.train()
    finally:
        if run is not None:
            run.finish()

    print("\nTesting...\n")

    # TESTING
    for weight_path in get_model_paths(cfg.path.weights_test, cfg.model.name):
        fp = FirePredict(cfg, weight_path)
        fp.test()


if __name__ == "__main__":
    init_seeds()
    main()
