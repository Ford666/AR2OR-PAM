from utils.utilfunc import *
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
from torchvision import transforms as T
import albumentations as A
from PIL import Image
import random
import imageio
import re

#data augmentation for ear MAP data + brain out-of-focus data

#ear training set: 1, 2, 3, 5, 6, 7, 9, 10, ear_1, ear_3, ear_4
#ear test: 4、8、ear_2

#brain training set: 1,2,3,4
#brain test: 0


class MAPDataset(Dataset):
    def __init__(self, x_dir, y_dir):
        super(MAPDataset, self).__init__()

        self.x_filenames = [join(x_dir, x) for x in listdir(x_dir)]
        self.y_filenames = [join(y_dir, y) for y in listdir(y_dir)]
        self.len = len(self.x_filenames)
        self.parms = []

        for p1 in range(2):  # filpping
            p1 = 0.5 if not p1 else p1
            for p2 in range(2):  # pad and randomcrop
                for p3 in range(2):  # random affine transform
                    for p4 in range(2):  # brightness & contrast
                        for p5 in range(2):  # elastic transform
                            parm = {'flip': p1, 'padcrop': p2, 'affine': p3,
                                    'illum': p4, 'et': p5}
                            if not p5:
                                self.parms.append(parm)
                            else:
                                for seed in [4, 42]:  # elastic transform
                                    parm['et_seed'] = seed
                                    self.parms.append(parm.copy())

    # get data operation
    def __getitem__(self, index):
        
        img_path = self.x_filenames[index]
        mask_path = self.y_filenames[index]
        img, mask = np.load(img_path), np.load(mask_path)
        img, mask = img/(np.amax(img)), mask/(np.amax(mask))    #self-normalization 
        [P_y, P_x] = img.shape

        
        # data augmentation via albumentations
        for j in range(len(self.parms)):
            trfm = A.Compose([
                A.Flip(p=self.parms[j]['flip']),
                A.Compose([
                    A.PadIfNeeded(p=1, min_height=int(6/5*P_y), 
                        min_width=int(6/5*P_x)),
                    A.RandomCrop(p=1, height=P_y, width=P_x)], p=self.parms[j]['padcrop']),
                A.ShiftScaleRotate(rotate_limit=[-15, 15], p=self.parms[j]['affine']),
                A.RandomGamma(gamma_limit=(60, 140), p=self.parms[j]['illum']),
                A.OneOf([
                    A.Blur(blur_limit=(30, 40)), 
                    A.GaussianBlur(blur_limit=(31,43), sigma_limit=(15, 17))], p=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.1, p=0.1),
                A.ElasticTransform(alpha=34, sigma=6,
                                   alpha_affine=160, p=self.parms[j]['et']),
            ])
            if self.parms[j]['et']:
                random.seed(self.parms[j]['et_seed'])
            augments = trfm(image=img, mask=mask)

            img_aug, mask_aug = augments['image'], augments['mask']

            img_name = re.findall(r'[^\\/:*?"<>|\r\n]+$', img_path) 
            img_aug_path = "../dataset/ear_AR_aug_v2/%s_%02d.png" % (img_name[0].replace(".npy", ""),j)

            mask_name = re.findall(r'[^\\/:*?"<>|\r\n]+$', mask_path) 
            mask_aug_path = "../dataset/ear_OR_aug_v2/%s_%02d.png" % (mask_name[0].replace(".npy", ""),j)
            print(j)
            imageio.imwrite(img_aug_path, (255*img_aug).astype(np.uint8))
            imageio.imwrite(mask_aug_path, (255*mask_aug).astype(np.uint8))

        return 0, 0

    def __len__(self):
        return self.len


def get_earDataloader():
    train_ds = MAPDataset('../dataset/Raw data/ear_MAP_processed_V2/train/ARv2',
                            '../dataset/Raw data/ear_MAP_processed_V2/train/ORv2')
    train_loader = DataLoader(dataset=train_ds, batch_size=1,
                              shuffle=False, num_workers=8)
    return train_loader

def get_brainDataloader():
    train_ds = MAPDataset('../dataset/Raw data/brain_out-of-focus_V2/train/AR',
                            '../dataset/Raw data/brain_out-of-focus_V2/train/OR')
    train_loader = DataLoader(dataset=train_ds, batch_size=1,
                              shuffle=False, num_workers=8)
    return train_loader
