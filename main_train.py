from utils.args import MAP_parse_args
from utils.utilfunc import *
from utils.dataloader import get_MAPdataloader, get_TestMAPdataloader
from TrainWGAN_GP import TrainWGAN_GP


if __name__ == "__main__":

    # arg
    opt = MAP_parse_args()

    # U-net architecture & loss function
    Unets = ['ResConvBlock', 'ResDenseBlock']
    loss_types = ['L1-SSIM', 'L1-FL1', 'L1-TV', 'L1-PCC', 'L1', 'L2']

    # Load train set
    train_loader = get_MAPdataloader(
        opt, (int(384*opt.scale), int(384*opt.scale)))

    # Load test set
    test_ear_groups = ["04", "08", "ear_2"]
    test_loaders = []
    for i in test_ear_groups:
        # The test image patches
        test_x_path, test_y_path = "%s/ear/%s/x" % (
            opt.test_dir, i), "%s/ear/%s/y" % (opt.test_dir, i)
        test_loaders.append(get_TestMAPdataloader(
            test_x_path, test_y_path, (int(384*opt.scale), int(384*opt.scale))))

    newH, newW = int(2000*opt.scale), int(2000*opt.scale)
    patchHW = int(384*opt.scale)
    overlap = int(64*opt.scale)

    # training WGAN-GP
    model = TrainWGAN_GP(opt, Unets[0], loss_types[0],
                         train_loader, test_loaders, test_ear_groups,
                         newH, newW, patchHW, overlap)

    model.train()
