import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from os.path import join
from os import listdir
from torchvision import transforms as T
import albumentations as A
from PIL import Image


# Custom Dataset class
class MAPDataset(Dataset):
    def __init__(self, x_dir, y_dir, IMAGE_SIZE):
        # def __init__(self, x_dir, y_dir):
        super(MAPDataset, self).__init__()

        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Resize(IMAGE_SIZE)
        ])

        self.x_filenames = [join(x_dir, x) for x in listdir(x_dir)]
        self.y_filenames = [join(y_dir, y) for y in listdir(y_dir)]
        self.len = len(self.x_filenames)

    def __getitem__(self, index):

        img = np.array(Image.open(self.x_filenames[index]))
        mask = np.array(Image.open(self.y_filenames[index]))

        # return F.normalize(self.as_tensor(img)), \
        #     F.normalize(self.as_tensor(mask))
        return self.as_tensor(img), self.as_tensor(mask)

    def __len__(self):
        return self.len


class TestMAPDataset(Dataset):
    # with label
    def __init__(self, x_dir, y_dir, IMAGE_SIZE):
        # def __init__(self, x_dir, y_dir):
        super(TestMAPDataset, self).__init__()

        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Resize(IMAGE_SIZE)
        ])

        self.x_filenames = [join(x_dir, x) for x in listdir(x_dir)]
        self.y_filenames = [join(y_dir, y) for y in listdir(y_dir)]
        self.len = len(self.x_filenames)

    def __getitem__(self, index):

        img = np.array(Image.open(self.x_filenames[index]))
        mask = np.array(Image.open(self.y_filenames[index]))

        return self.as_tensor(img), self.as_tensor(mask)

    def __len__(self):
        return self.len


def get_MAPdataloader(opt, IMAGE_SIZE):

    dataset = MAPDataset(opt.train_x_path, opt.train_y_path, IMAGE_SIZE)

    train_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size,
                              shuffle=True, num_workers=0)
    return train_loader


def get_TestMAPdataloader(test_x_path, test_y_path, IMAGE_SIZE):
    # with label
    test_ds = TestMAPDataset(test_x_path, test_y_path, IMAGE_SIZE)
    test_loader = DataLoader(dataset=test_ds, batch_size=1,
                             shuffle=False, num_workers=0)
    return test_loader
