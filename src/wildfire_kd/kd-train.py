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

from src.wildfire_kd.datasets import Landset8Dataset, Landset8ManualDataset
from src.wildfire_kd.model_manager import ModelManager
from src.wildfire_kd.utils import *
from src.wildfire_kd.adapter import MLP

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
        self.criterion_KD = nn.MSELoss()

        self.adapter = nn.Conv2d(128, 256, kernel_size=1, bias=False)

        self.optimizer = optim.Adam(
            list(self.student.parameters()) + list(self.adapter.parameters()),
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
            train_dataset, batch_size=self.cfg.const.batch_size, shuffle=True, num_workers=4, drop_last=True,
            prefetch_factor=8, persistent_workers=True,
        )

        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.cfg.const.batch_size, shuffle=False, num_workers=4, drop_last=True,
            prefetch_factor=8, persistent_workers=True,
        )


    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher = self.teacher.to(device)
        self.student = self.student.to(device)
        self.adapter = self.adapter.to(device)

        best_val_loss = float('inf')
        epsilon = 1e-14
        num_epochs = self.cfg.const.num_epochs
        T = self.cfg.distill.T
        alpha = self.cfg.distill.alpha
        lbd = self.cfg.distill.lbd

        # wandb.watch(self.student, log="all", log_freq=100)

        for epoch in range(num_epochs):
            self.student.train()
            running_loss = 0.0
            running_loss_CEL = 0.0
            running_loss_KD = 0.0

            for images, labels in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # for images, labels in self.train_dataloader:
                images = images.to(device)
                labels = labels.unsqueeze(1).float().to(device)

                with torch.no_grad():
                    teacher_out = self.teacher(images)
                student_out = self.student(images)

                teacher_logit = teacher_out['out']
                student_logit = student_out['out']

                loss_CEL = self.criterion(student_logit, labels)

                # KD loss
                s_feat = student_out['aux']
                t_feat = teacher_out['aux']

                if t_feat.shape[-2:] != s_feat.shape[-2:]:
                    s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)

                s_feat = self.adapter(s_feat) 
                
                loss_KD = self.criterion_KD(s_feat, t_feat)

                # print(float(loss_KD.detach()))

                loss = loss_CEL + lbd * loss_KD

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_loss_CEL += loss_CEL.item()
                running_loss_KD += loss_KD.item()

            epoch_loss = running_loss / len(self.train_dataloader)
            epoch_loss_CEL = running_loss_CEL / len(self.train_dataloader)
            epoch_loss_KD = running_loss_KD / len(self.train_dataloader)

            # wandb.log({"train_loss": epoch_loss, "train_loss_CEL": epoch_loss_CEL, "train_loss_KD": epoch_loss_KD})
            print(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.7f}, CEL: {epoch_loss_CEL:.7f}, KD: {epoch_loss_KD:.7f}')

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
            images_csv="./data/split/images_test.csv",
            dataset_dir=self.cfg.path.dataset,
            mask=self.cfg.const.mask,
            augment=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.cfg.const.batch_size, shuffle=False, num_workers=8, drop_last=True,
            prefetch_factor=8, persistent_workers=True,
        )

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


    def test_manual(self):
        test_dataset = Landset8ManualDataset(
            root_dir=self.cfg.path.dataset_manual
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.cfg.const.batch_size, shuffle=False, num_workers=8, drop_last=True,
            prefetch_factor=8, persistent_workers=True,
        )

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


    def get_efficiency(self):
        print(torch.cuda.get_device_name(torch.cuda.current_device()) + "\n")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.student.to(device).eval()

        images, _ = self.val_dataloader.dataset[3]
        x = images.unsqueeze(0).to(device, non_blocking=True)

        print(self.cfg.model.name, end='\t')

        # Params (M)
        print(sum(p.numel() for p in model.parameters()) / 1e6, end='\t')

        # Size (MB)
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('{:.3f}MB'.format(size_all_mb), end='\t')

        # FLOPS (G) fvcore
        with torch.inference_mode():
            from fvcore.nn import FlopCountAnalysis

            class Wrapper(torch.nn.Module): # FlopCountAnalysis expects olny out.
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, x):
                    out = self.model(x)
                    return out["out"] if isinstance(out, dict) and "out" in out else out

            wrapped = Wrapper(model).eval()
            flops = FlopCountAnalysis(wrapped, x)
            total_flops = flops.total()
            print('{:.3f}G'.format(total_flops / 1e9), end='\t')

        # FLOPS (G) OpCounter
        with torch.inference_mode():
            from thop import profile, clever_format
            macs_thop, params_thop = profile(wrapped, inputs=(x,), verbose=False)
            flops = (2 * macs_thop) / 1e9

            macs_str, params_str = clever_format([macs_thop, params_thop], "%.3f")
            print(f"THOP: {macs_str} MACs ({flops:.3f} GFLOPs)")

        # FLOPS (G) torchinfo
        total_params_ti = None
        total_madds_ti = None
        from torchinfo import summary
        stats = summary(
            wrapped,
            input_data=(x,),
            device=device,
            mode="eval",
            col_names=("output_size", "num_params", "mult_adds"),
            verbose=0,
        )
        total_params_ti = getattr(stats, "total_params", None)
        total_madds_ti = getattr(stats, "total_mult_adds", None)

        if total_madds_ti is not None:
            gmacs_ti = total_madds_ti / 1e9
            gflops_ti = 2 * gmacs_ti
            print(f"torchinfo: {gmacs_ti:.3f} GMAdd ({gflops_ti:.3f} GFLOPs)", end='\t')
        if total_params_ti is not None:
            print(f"params={total_params_ti/1e6:.3f}M")

        # Time (ms) batch=1
        torch.backends.cudnn.benchmark = True
        torch.cuda.synchronize()

        with torch.inference_mode():    # warmup
            for _ in range(500):
                _ = model(x)
        torch.cuda.synchronize()
        times = []
        with torch.inference_mode():    # timed
            for _ in range(20000):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                _ = model(x)
                end.record()

                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))  # ms
        mean_ms = float(np.mean(times))
        print(f"{mean_ms:.2f}ms")


@hydra.main(config_path="../../conf", config_name="kd", version_base=None)
def main(cfg: DictConfig):
    # wandb.init(project="FirePrediction", config=OmegaConf.to_container(cfg, resolve=True), name=f"Run_{time.strftime('%Y%m%d-%H%M%S')}")
    # print("Configuration: ")
    # print(OmegaConf.to_yaml(cfg))

    # TRAINING
    # fp = FirePredict(cfg)
    # fp.train()
    # wandb.finish()

    # print("\nTesting...\n")

    # TESTING
    # for weight_path in get_model_paths(cfg.path.weights_test, cfg.model.name):
    #     fp = FirePredict(cfg, weight_path)
    #     fp.test()

    cfg.model.name = "unet"
    FirePredict(cfg).get_efficiency()

    cfg.model.name = "enet"
    FirePredict(cfg).get_efficiency()

    cfg.model.name = "fddwnet"
    FirePredict(cfg).get_efficiency()


if __name__ == "__main__":
    init_seeds()
    main()
