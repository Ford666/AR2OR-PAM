from MAP_aug import get_earDataloader, get_brainDataloader
from utils.utilfunc import *


if __name__ == "__main__":

    ear_train_loader = get_earDataloader()
    for imgs, masks in progressbar(ear_train_loader):
        None

    # brain_train_loader = get_brainDataloader()
    # for imgs, masks in progressbar(brain_train_loader):
    #     None
