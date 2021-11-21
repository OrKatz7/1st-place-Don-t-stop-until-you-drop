# 1st-place-Don't-stop-until-you-drop
# TL;DR
## model - swin transform (large and base) image size 384
## 5-fold class blanced
## soft transforms
'''

def get_transforms(*, data):
    
    if data == 'train':
        return Compose([
            Resize(CFG.size, CFG.size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.RandomContrast(limit=0.2, p=0.75),
            HorizontalFlip(p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
            albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=5, border_mode=0, p=0.75),
            CutoutV2(max_h_size=int(CFG.size * 0.2), max_w_size=int(CFG.size * 0.2), num_holes=1, p=0.75),# VerticalFlip(p=0.5),
            # ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
'''
