import wandb
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
import os

from landset8_loader import Landset8Dataset
from model_manager import ModelManager
from utils import *

# -----------------------------
#   Stability options
# -----------------------------
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('medium')

# -----------------------------
#   WandB login
# -----------------------------
with open("wandb_secret.json", "r") as f:
    secrets = json.load(f)
wandb.login(key=secrets["WANDB_API_KEY"])

# ==============================================================
#   FIREPREDICT KD FINE-TUNE
# ==============================================================

class FirePredict:
    def __init__(self, cfg: DictConfig, finetune_ckpt=None):
        self.cfg = cfg
        self._create_data_loaders()

        os.makedirs(self.cfg.path.output_model, exist_ok=True)

        # --- Teacher (frozen) ---
        self.teacher = ModelManager(cfg, cfg.teacher.name.lower()).create_model()
        self.teacher.load_state_dict(torch.load(cfg.path.teacher_model, map_location="cpu"))
        self.teacher.eval()

        # --- Student (pruned checkpoint) ---
        self.student = ModelManager(cfg, cfg.model.name.lower()).create_model()
        if finetune_ckpt is not None:
            print(f"Loading student fine-tune checkpoint from: {finetune_ckpt}")
            self.student.load_state_dict(torch.load(finetune_ckpt, map_location="cpu"))
            self.checkpoint_path = finetune_ckpt
        else:
            print("⚠️ No fine-tune checkpoint provided — starting student from scratch.")
            self.checkpoint_path = "scratch"

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.student.parameters(),
            lr=self.cfg.const.lr,
            weight_decay=self.cfg.const.weight_decay,
        )

    # -----------------------------
    #   Data loaders
    # -----------------------------
    def _create_data_loaders(self):
        train_dataset = Landset8Dataset(
            images_csv="./split/images_train.csv",
            dataset_dir=self.cfg.path.dataset,
            mask=self.cfg.const.mask,
            augment=True,
        )
        val_dataset = Landset8Dataset(
            images_csv="./split/images_val.csv",
            dataset_dir=self.cfg.path.dataset,
            mask=self.cfg.const.mask,
            augment=False,
        )

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.cfg.const.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.cfg.const.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=True
        )

    # ==============================================================
    #   TRAIN LOOP — KNOWLEDGE DISTILLATION FINETUNE
    # ==============================================================
    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = self.teacher.to(device)
        self.student = self.student.to(device)

        run_name = f"KD_Finetune_{self.cfg.model.name}_{time.strftime('%Y%m%d-%H%M%S')}"
        wandb.init(project="FirePrediction_Finetune",
                   config=OmegaConf.to_container(self.cfg, resolve=True),
                   name=run_name)
        wandb.watch(self.student, log="all", log_freq=100)

        best_val_loss = float("inf")
        epsilon = 1e-14
        num_epochs = self.cfg.const.num_epochs
        T = self.cfg.distill.T
        alpha = self.cfg.distill.alpha
        beta = self.cfg.distill.beta
        eps = 1e-6

        for epoch in range(num_epochs):
            self.student.train()
            running_loss = 0.0

            for step, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(device), labels.unsqueeze(1).float().to(device)

                # --- Teacher forward (frozen) ---
                with torch.no_grad():
                    teacher_logit = self.teacher(images)["out"]

                # --- Student forward ---
                student_logit = self.student(images)["out"]

                # --- Standard BCE loss ---
                loss_CE = self.criterion(student_logit, labels)

                # --- Distillation loss (stable form) ---
                with torch.no_grad():
                    t_soft = torch.sigmoid(teacher_logit / T).clamp(eps, 1 - eps)

                loss_KD = F.binary_cross_entropy_with_logits(
                    student_logit / T, t_soft, reduction="mean"
                ) * (T * T)

                # --- Total loss ---
                loss = alpha * loss_CE + beta * (1 - alpha) * loss_KD

                # Safety guard
                if not torch.isfinite(loss):
                    print(f"[NaN detected] Epoch={epoch} Step={step}")
                    continue

                
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
                self.optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(self.train_dataloader)
            wandb.log({"train_loss": train_loss})
            print(f"Epoch {epoch+1}/{num_epochs} Training loss: {train_loss:.6f}")

            # --- Validation ---
            val_loss, precision, recall, f1, iou, acc = self.validate(device, epsilon)
            print(f"Validation Loss: {val_loss:.5f}, "
                  f"Precision: {precision:.3f}, Recall: {recall:.3f}, "
                  f"F1: {f1:.3f}, IoU: {iou:.3f}, Acc: {acc:.3f}")

            wandb.log({
                "val_loss": val_loss, "val_precision": precision,
                "val_recall": recall, "val_f1": f1, "val_acc": acc,
            })

            # --- Save best checkpoint ---
            ckpt_name = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
            save_dir = os.path.join(self.cfg.path.output_model, ckpt_name)
            os.makedirs(save_dir, exist_ok=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(
                    save_dir,
                    f"{self.cfg.model.name}_finetune_{timestamp}_ep{epoch}_val{val_loss:.3f}_f1{f1:.3f}.pt"
                )
                torch.save(self.student.state_dict(), save_path)
                print(f"✅ Saved improved model: {save_path}")

        wandb.finish()

    # ==============================================================
    #   VALIDATION LOOP
    # ==============================================================
    def validate(self, device, epsilon):
        self.student.eval()
        val_loss = 0.0
        TP_all = np.zeros((2, 1))
        FP_all = np.zeros((2, 1))
        TN_all = np.zeros((2, 1))
        FN_all = np.zeros((2, 1))

        with torch.no_grad():
            for images, labels in self.val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.student(images)["out"]

                labels_float = labels.unsqueeze(1).float()
                val_loss += self.criterion(outputs, labels_float).item()

                preds = (outputs[:, 0, :, :] > self.cfg.const.th_fire).cpu().numpy().astype(np.uint8).flatten()
                labels_np = labels.cpu().numpy().flatten()
                TP, FP, TN, FN, _ = eval_image(preds, labels_np, 2)
                TP_all += TP; FP_all += FP; TN_all += TN; FN_all += FN

        val_loss /= len(self.val_dataloader)
        acc = (TP_all[1] + TN_all[1]) / (TP_all[1] + TN_all[1] + FP_all[1] + FN_all[1] + epsilon)
        precision = TP_all[1] / (TP_all[1] + FP_all[1] + epsilon)
        recall = TP_all[1] / (TP_all[1] + FN_all[1] + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = TP_all[1] / (TP_all[1] + FP_all[1] + FN_all[1] + epsilon)
        return val_loss, precision[0], recall[0], f1[0], iou[0], acc[0]


# ==============================================================
#   HYDRA ENTRY POINT
# ==============================================================
@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    ckpt_paths = get_model_paths(cfg.path.weights_finetune, cfg.model.name)
    print(f"\nFound {len(ckpt_paths)} checkpoints:", [os.path.basename(p) for p in ckpt_paths])

    for ckpt in ckpt_paths:
        print(f"\n==== KD Fine-tuning from checkpoint: {ckpt} ====\n")
        fp = FirePredict(cfg, finetune_ckpt=ckpt)
        fp.train()


if __name__ == "__main__":
    init_seeds()
    main()
