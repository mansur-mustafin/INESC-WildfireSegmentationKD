# Description

same as previous, bus has diferent data augmentation technics:

Old:
```py
self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            ])
```

New:
```py
self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0, p=0.5),
                A.RandomResizedCrop(width=512, height=512, scale=(0.8, 1.0), ratio=(1.0, 1.0), p=0.5),
                A.RandomRotate90(p=0.5),
            ])
```
