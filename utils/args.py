import argparse
import os


def MAP_parse_args():
    parser = argparse.ArgumentParser(
        description="Hyper parameter of WGAN_GP model.")

    parser.add_argument('--batch_size', type=int,
                        default=8, help='The size of batch')

    parser.add_argument('--epoches', type=int, default=12,
                        help='The number of epoches to run')
    parser.add_argument('--G_lr', type=float,
                        default=1e-4, help='The initial Lr of G')
    parser.add_argument('--D_lr', type=float,
                        default=1e-4, help='The initial Lr of D')
    parser.add_argument('--weight_clipping_limit', type=float, default=0.01)
    parser.add_argument('--resume', type=int, default=0,
                        choices=[0, 1], help='Whether to resume training via checkpoint')

    parser.add_argument('--train_result_path', type=str,
                        default='../result/training/2021 Oct 28_5/')
    parser.add_argument('--model_dir', type=str,
                        default='./checkpoints')

    parser.add_argument('--train_x_path', type=str,
                        default='../dataset/ARimgP_aug/ear_v3')
    parser.add_argument('--train_y_path', type=str,
                        default='../dataset/ORimgP_aug/ear_v3')
    parser.add_argument('--test_dir', type=str,
                        default='../result/test')
    parser.add_argument('--scale', type=float,
                        default=1, help='scale image data for training')
    return parser.parse_args()


if __name__ == '__main__':
    args = MAP_parse_args()
    print(args)
