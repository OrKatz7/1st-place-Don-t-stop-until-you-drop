# 1st-place-Don't-stop-until-you-drop
This repository represents open-source research developed by [Seffi Cohen](https://www.linkedin.com/in/seffi-cohen-11182046/), [Niv Goldshlager](https://www.linkedin.com/in/niv-goldshlager/), [Nurit Cohen Inger](https://www.linkedin.com/in/nurit-cohen-inger-265269b2/) and [Or Katz](https://www.linkedin.com/in/or-katz-9ba885114/) ,  for the 1st place solution to the kaggle days championship - Don't stop until you drop!

## train + inferance
1. run train_p2_swin_large_patch4_window12_384.ipynb
2. run train_p2_swin_base_patch4_window12_384.ipynb
3. run inferance_swin_large_patch4_window12_384-final.ipynb
# TL;DR

1. swin transform (large and base) image size 384
2. 5-fold class blanced
3. scheduler - 'CosineAnnealingWarmRestarts'
4. soft transforms
```` 
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
```` 
5. ensemble swin-l and swin-b (4 folds from l and 2 from b)
6. config
```` 
class CFG:
    debug=False
    apex=False
    print_freq=100
    num_workers=8
    model_name='swin_large_patch4_window12_384'
    size=384
    scheduler='CosineAnnealingWarmRestarts' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs=8
    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    #T_max=10 # CosineAnnealingLR
    T_0=10 # CosineAnnealingWarmRestarts
    lr=1e-4
    min_lr=1e-6
    batch_size=12
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    target_size=6
    target_col='class_6'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True
    inference=False
```` 
