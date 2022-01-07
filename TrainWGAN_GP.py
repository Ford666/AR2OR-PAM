# -*- coding: utf8 -*-
from utils.utilfunc import *
from models.GANmodel import *
from models.loss import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrainWGAN_GP(object):
    def __init__(self, opt, Unet_type, loss_type,
                 train_loader, test_loaders, test_ear_groups,
                 newH, newW, patchHW, overlap):

        print("WGAN with gradient penalty")

        # training and checkpoint path
        self.train_result_path = opt.train_result_path
        self.SR_path, self.AR_path, self.OR_path =  \
            self.train_result_path + 'SR_iter/', self.train_result_path + \
            'AR_iter/', self.train_result_path + 'OR_iter/'
        self.test_SR_path = self.train_result_path + 'test_SR_iter/'
        self.test_ear_groups = test_ear_groups
        self.H, self.W = newH, newW

        mkdir(self.SR_path)
        mkdir(self.AR_path)
        mkdir(self.OR_path)
        mkdir(self.test_SR_path)

        self.file = open(
            '{}/Trainfile.txt'.format(self.train_result_path), 'w+')

        self.model_dir = '%s/%s_%s' % (opt.model_dir, Unet_type, loss_type)
        mkdir(self.model_dir)

        if opt.resume:
            epoch = input("which model to load: ")
            self.model_path = self.model_dir + "/MAP_model_%s.pth.tar" % epoch

        self.epoches = opt.epoches
        self.resume = opt.resume

        # Define model, optimizer
        self.G = Generator(Unet_type).cuda()
        self.D = Discriminator().cuda()
        init_weights(self.G, scale=0.1)
        init_weights(self.D, scale=1)

        self.G_lr, self.D_lr = opt.G_lr, opt.D_lr

        self.optimizerD = torch.optim.AdamW(self.D.parameters(),
                                            lr=self.G_lr, betas=(0.5, 0.9))
        self.optimizerG = torch.optim.AdamW(self.G.parameters(),
                                            lr=self.D_lr, betas=(0.5, 0.9))
        self.schedulerG = lr_scheduler.CosineAnnealingLR(
            self.optimizerG, T_max=self.epoches, eta_min=0)
        self.schedulerD = lr_scheduler.CosineAnnealingLR(
            self.optimizerD, T_max=self.epoches, eta_min=0)

        # self.weight_clipping_limit = opt.weight_clipping_limit

        # training records
        self.epoch_result = {'epoch': [], 'lossD': [], 'lossG': [],
                             'train_Gloss': [], 'test_Gloss': []}
        self.iter_result = {'D_fake': [], 'D_real': [], 'lossD': [],
                            'lossG': [], 'train_Gloss': []}
        self.test_metrics = {'epoch': [], 'AR_PSNR': [], 'AR_SSIM': [], 'AR_PCC': [],
                             'SR_PSNR': [], 'SR_SSIM': [], 'SR_PCC': []}

        self.loss_type = loss_type

        # train and test dataset
        self.train_loader = train_loader
        self.test_loaders = test_loaders
        self.patchHW, self.overlap = patchHW, overlap

    def train(self):

        start_epoch = 1
        iters = 0       # total number of iterations
        self.data_iter = iter_batches(self.train_loader)  # data iterator
        ITER_PER_EPOCH = len(self.train_loader)

        # load checkpoint to continue training
        if self.resume:
            assert os.path.isfile(self.model_path)
            checkpoint = torch.load(self.model_path)
            self.D.load_state_dict(checkpoint['D_model'])
            self.G.load_state_dict(checkpoint['G_model'])
            self.optimizerD.load_state_dict(checkpoint['optimizerD'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG'])

            start_epoch = checkpoint['start_epoch']
            iters = checkpoint['iters']
            self.epoch_result = checkpoint['epoch_result']
            self.iter_result = checkpoint['iter_result']
            self.test_metrics = checkpoint['test_metrics']
            print("Load epoch {} done!".format(start_epoch-1))

        for epoch in np.arange(start_epoch, self.epoches+1):
            start_time = time.time()
            self.G.train()
            self.D.train()

            for i in progressbar(np.arange(iters % ITER_PER_EPOCH, ITER_PER_EPOCH)):

                iters += 1
                # train D several times in the first
                # once D is trained strong, we set adaptive Diters and Giters
                if epoch <= 4:
                    if epoch % 2 == 1:
                        Diters, Giters = 3, 1
                    else:
                        Diters, Giters = 1, 2
                else:
                    if iters % 400 == 0:
                        Diters, Giters = 3, 1
                    else:
                        Diters, Giters = 1, 1
                ############################
                # (1) Update D network
                ###########################
                for p in self.D.parameters():
                    p.requires_grad = True

                for j in range(Diters):
                    self.optimizerD.zero_grad()

                    # Instead of iterating on the same data-label, we use iterative data-label per Diter
                    (x, y) = self.data_iter.__next__()
                    AR_img, OR_img = x.to(DEVICE), y.to(DEVICE)
                    AR_img.requires_grad_(True)
                    OR_img.requires_grad_(True)

                    # detached from computeration graph
                    SR_img = self.G(AR_img).data

                    # train with real data
                    D_real = self.D(OR_img)

                    # train with fake data
                    D_fake = self.D(SR_img)

                    # train with gradient penalty
                    GP = 10 * GradPenalty(self.D, OR_img, SR_img)

                    # Wasserstein GAN Loss
                    lossD = D_fake.mean() - D_real.mean() + GP

                    #D_fake.mean(0).view(1).backward(one) or GP.backward(one)
                    lossD.backward()
                    self.optimizerD.step()

                # make records
                self.iter_result['D_fake'].append(D_fake.mean().item())
                self.iter_result['D_real'].append(D_real.mean().item())
                self.iter_result['lossD'].append(lossD.item())

                ############################
                # (2) Update G network
                ###########################
                for p in self.D.parameters():
                    p.requires_grad = False

                # train G several times in the next few epoches
                for j in range(Giters):
                    self.optimizerG.zero_grad()

                    # Instead of iterating on the same data-label, we use iterative data-label per Diter
                    (x, y) = self.data_iter.__next__()
                    AR_img, OR_img = x.to(DEVICE), y.to(DEVICE)
                    AR_img.requires_grad_(True)
                    OR_img.requires_grad_(True)

                    SR_img = self.G(AR_img)
                    D_fake = self.D(SR_img)

                    # LossG
                    Gloss = Gloss_fn(self.loss_type, SR_img, OR_img)
                    lossG_adv = -1e-3 * D_fake.mean()
                    lossG = Gloss[-1] + lossG_adv

                    # Computes the sum of gradients of given tensors w.r.t. graph leaves
                    lossG.backward()
                    self.optimizerG.step()

                # make records
                self.iter_result['train_Gloss'].append(Gloss[-1].item())
                self.iter_result['lossG'].append(lossG.item())

                # Save training results
                if iters % 400 == 0 or iters % ITER_PER_EPOCH == 0:
                    saveImgfromTensor(SR_img, self.SR_path, iters)
                    saveImgfromTensor(AR_img, self.AR_path, iters)
                    saveImgfromTensor(OR_img, self.OR_path, iters)
                    self.plot_iter_result()
                    self.file.write("cur.epoch: %d/%d, cur.iter: %d/%d, \
D_real: %.6f, D_fake: %.6f, train_L1loss: %.6f\n" % (
                        epoch, self.epoches, iters,
                        self.epoches * ITER_PER_EPOCH,
                        np.mean(np.array(self.iter_result['D_real'])),
                        np.mean(np.array(self.iter_result['D_fake'])),
                        np.mean(np.array(self.iter_result['train_Gloss'])),
                    ))
                    self.file.flush()

            self.epoch_result['epoch'].append(epoch)
            self.epoch_result['lossD'].append(
                np.mean(np.array(self.iter_result['lossD'])))
            self.epoch_result['lossG'].append(
                np.mean(np.array(self.iter_result['lossG'])))
            self.epoch_result['train_Gloss'].append(
                np.mean(np.array(self.iter_result['train_Gloss'])))

            # update the learning rate per epoch
            self.schedulerD.step()
            self.schedulerG.step()

            ############# Test per epoch##################
            test_records = self.test(epoch)
            self.epoch_result['test_Gloss'].append(test_records[0])
            self.test_metrics['epoch'].append(epoch)
            self.test_metrics['AR_PSNR'].append(test_records[1])
            self.test_metrics['AR_SSIM'].append(test_records[2])
            self.test_metrics['AR_PCC'].append(test_records[3])
            self.test_metrics['SR_PSNR'].append(test_records[4])
            self.test_metrics['SR_SSIM'].append(test_records[5])
            self.test_metrics['SR_PCC'].append(test_records[6])

            print("cur.epoch: %d, D_real: %.4f, D_fake: %.4f, train_L1loss: %.4f, test_Gloss: %.4f, time used: %.4f min\n" % (
                epoch, np.mean(np.array(self.iter_result['D_real'])), np.mean(
                    np.array(self.iter_result['D_fake'])),
                self.epoch_result['train_Gloss'][-1], self.epoch_result['test_Gloss'][-1], (time.time()-start_time)/60))

            self.plot_epoch_result()
            self.plot_test_metrics()

            ############# Save checkpoint per epoch ################

            self.model_path = self.model_dir + "/MAP_model_%s.pth.tar" % epoch
            checkpoint = {
                'start_epoch': epoch+1,
                'epoch_result': self.epoch_result,
                'iter_result': self.iter_result,
                'test_metrics': self.test_metrics,
                'iters': iters,
                'G_model': self.G.state_dict(),
                'D_model': self.D.state_dict(),
                'optimizerG': self.optimizerG.state_dict(),
                'optimizerD': self.optimizerD.state_dict(),
            }
            torch.save(checkpoint, self.model_path)

        self.file.close()

    def test(self, epoch):
        test_Gloss = []
        ARpsnrs, ARssims, ARpccs = [], [], []
        SRpsnrs, SRssims, SRpccs = [], [], []

        with torch.no_grad():
            self.G.eval()
            self.D.eval()
            ind = 0
            for test_loader in self.test_loaders:
                SR_imgs = torch.Tensor([]).cuda()
                for image, target in progressbar(test_loader):
                    AR_img, OR_img = image.to(DEVICE), target.to(DEVICE)
                    SR_img = self.G(AR_img)
                    SR_imgs = torch.cat(
                        (SR_imgs, SR_img.clone().squeeze(1)), 0)

                    # Gloss
                    Gloss = Gloss_fn(self.loss_type, SR_img, OR_img)
                    test_Gloss.append(Gloss[-1].item())

                    # metrics
                    AR_img, SR_img, OR_img = torch.squeeze(AR_img).data.cpu().numpy(), torch.squeeze(
                        SR_img).data.cpu().numpy(), torch.squeeze(OR_img).data.cpu().numpy()
                    ARpsnrs.append(calculate_psnr(AR_img, OR_img))
                    SRpsnrs.append(calculate_psnr(SR_img, OR_img))
                    ARssims.append(calculate_ssim(AR_img, OR_img))
                    SRssims.append(calculate_ssim(SR_img, OR_img))
                    ARpccs.append(calculate_pcc(AR_img, OR_img))
                    SRpccs.append(calculate_pcc(SR_img, OR_img))

                stitch_patch_V1(self.test_SR_path + 'SR%s_epoch%s' % (self.test_ear_groups[ind], epoch),
                                SR_imgs.data.cpu().numpy(), self.H, self.W,
                                self.patchHW, self.patchHW, self.overlap)
                ind += 1

            self.file.write("AR|OR patches, PSNR: %.6f, SSIM: %.6f, PCC: %.6f\n" % (
                np.array(ARpsnrs).mean(), np.array(ARssims).mean(), np.array(ARpccs).mean()))
            self.file.write("SR|OR patches, PSNR: %.6f, SSIM: %.6f, PCC: %.6f\n" % (
                np.array(SRpsnrs).mean(), np.array(SRssims).mean(), np.array(SRpccs).mean()))
            self.file.flush()

        return np.array(test_Gloss).mean(), np.array(ARpsnrs).mean(), np.array(ARssims).mean(), np.array(ARpccs).mean(), np.array(SRpsnrs).mean(), np.array(SRssims).mean(), np.array(SRpccs).mean()

    def plot_iter_result(self):

        iters = len(self.iter_result["D_fake"])
        iter_list = [i*4 for i in range(iters)]  # case for batch_size=2

        D_fake = self.iter_result["D_fake"]
        D_real = self.iter_result["D_real"]
        lossD = self.iter_result["lossD"]

        L1loss = self.iter_result["train_Gloss"]
        lossG = self.iter_result["lossG"]

        # historial mean
        D_fake_mean = np.array([sum(D_fake[:idx])/(idx+1) for
                                idx in range(iters)])
        D_real_mean = np.array([sum(D_real[:idx])/(idx+1) for
                                idx in range(iters)])
        lossG_mean = np.array([sum(lossG[:idx])/(idx+1) for
                               idx in range(iters)])
        lossD_mean = np.array([sum(lossD[:idx])/(idx+1) for
                               idx in range(iters)])

        fig1, fig2 = plt.figure(1, figsize=(
            18, 5)), plt.figure(2, figsize=(12, 5))
        ax11, ax12, ax13 = fig1.add_subplot(
            131), fig1.add_subplot(132), fig1.add_subplot(133)
        ax21, ax22 = fig2.add_subplot(121), fig2.add_subplot(122)

        ax11.yaxis.grid(True, linestyle='-')
        ax11.plot(iter_list, D_real_mean,
                  'k-',  label="D(x)", linewidth=2)
        ax11.plot(iter_list, D_fake_mean,
                  'b-',  label="D(G(z))", linewidth=2)

        ax11.set_xlabel('Iterations', fontsize=16)
        ax11.set_ylabel('D_train', fontsize=16)
        ax11.legend(loc='upper right')

        ax12.plot(iter_list, D_fake_mean - D_real_mean, linewidth=2)
        ax12.set_xlabel('Iterations', fontsize=16)
        ax12.set_ylabel('WD', fontsize=16)
        ax13.plot(iter_list, lossD_mean, linewidth=2)
        ax13.set_xlabel('Iterations', fontsize=16)
        ax13.set_ylabel('train_lossD', fontsize=16)

        ax21.plot(iter_list, L1loss, linewidth=2)
        ax21.set_xlabel('Iterations', fontsize=16)
        ax21.set_ylabel('G_L1loss', fontsize=16)

        ax22.plot(iter_list, lossG_mean, linewidth=2)
        ax22.set_xlabel('Iterations', fontsize=16)
        ax22.set_ylabel('train_lossG', fontsize=16)

        # save the figure to file
        fig1.savefig('%s/train_lossD.png' % self.train_result_path,
                     dpi=300, bbox_inches='tight')
        fig2.savefig('%s/train_lossG.png' %
                     self.train_result_path, dpi=300, bbox_inches='tight')
        plt.close("all")
        plt.pause(2)

    def plot_epoch_result(self):
        epoches = self.epoch_result["epoch"]
        lossD = self.epoch_result['lossD']
        lossG = self.epoch_result['lossG']
        train_L1loss = self.epoch_result['train_Gloss']
        test_Gloss = self.epoch_result['test_Gloss']

        fig1, fig2 = plt.figure(1), plt.figure(2, figsize=(12, 5))
        ax1 = fig1.add_subplot(111)
        ax21, ax22 = fig2.add_subplot(121), fig2.add_subplot(122)
        xticks = np.rint(np.linspace(0, len(epoches), 5)).astype(int)

        ax1.yaxis.grid(True, linestyle='-')
        ax1.plot(epoches, train_L1loss,
                 'g-', label="train_L1loss", linewidth=2)
        ax1.plot(epoches, test_Gloss,
                 'p-', label="test_L1loss", linewidth=2)
        ax1.xaxis.set_major_locator(MultipleLocator(2))  # per 2 epoches
        ax1.set_xlabel('Epoches', fontsize=16)
        ax1.set_ylabel('L1loss', fontsize=16)
        ax1.legend(loc='upper right')

        ax21.yaxis.grid(True, linestyle='-')
        ax21.plot(epoches, lossG, linewidth=2)
        ax21.xaxis.set_major_locator(MultipleLocator(2))  # per 2 epoches
        ax21.set_xlabel('Epoches', fontsize=16)
        ax21.set_ylabel('LossG', fontsize=16)

        ax22.yaxis.grid(True, linestyle='-')
        ax22.plot(epoches, lossD, linewidth=2)
        ax22.xaxis.set_major_locator(MultipleLocator(2))  # per 2 epoches
        ax22.set_xlabel('Epoches', fontsize=16)
        ax22.set_ylabel('LossD', fontsize=16)

        fig1.savefig('%s/Epoch_L1loss.png' %
                     self.train_result_path, dpi=300, bbox_inches='tight')
        fig2.savefig('%s/Epoch_lossG_lossD.png' %
                     self.train_result_path, dpi=300, bbox_inches='tight')

        plt.close("all")
        plt.pause(2)

    def plot_test_metrics(self):
        epoches = self.test_metrics["epoch"]
        test_ARpsnr, test_ARssim, test_ARpcc = self.test_metrics[
            'AR_PSNR'], self.test_metrics['AR_SSIM'], self.test_metrics['AR_PCC']
        test_SRpsnr, test_SRssim, test_SRpcc = self.test_metrics[
            'SR_PSNR'], self.test_metrics['SR_SSIM'], self.test_metrics['SR_PCC']

        fig1 = plt.figure(1, figsize=(18, 5))
        ax1, ax2, ax3 = fig1.add_subplot(
            131), fig1.add_subplot(132), fig1.add_subplot(133)
        xticks = np.rint(np.linspace(0, len(epoches), 5)).astype(int)

        ax1.yaxis.grid(True, linestyle='-')
        ax1.plot(epoches, test_ARpsnr, 'k-', label="ARpsnr", linewidth=2)
        ax1.plot(epoches, test_SRpsnr, 'r-', label="SRpsnr", linewidth=2)
        ax1.xaxis.set_major_locator(MultipleLocator(2))  # per 2 epoches
        ax1.set_xlabel('Epoches', fontsize=16)
        ax1.set_ylabel('PSNR on test set', fontsize=16)
        ax1.legend(loc='upper right')

        ax2.yaxis.grid(True, linestyle='-')
        ax2.plot(epoches, test_ARssim, 'k-', label="ARssim", linewidth=2)
        ax2.plot(epoches, test_SRssim, 'r-', label="SRssim", linewidth=2)
        ax2.xaxis.set_major_locator(MultipleLocator(2))  # per 2 epoches
        ax2.set_xlabel('Epoches', fontsize=16)
        ax2.set_ylabel('SSIM on test set', fontsize=16)
        ax2.legend(loc='upper right')

        ax3.yaxis.grid(True, linestyle='-')
        ax3.plot(epoches, test_ARpcc, 'k-', label="ARpcc", linewidth=2)
        ax3.plot(epoches, test_SRpcc, 'r-', label="SRpcc", linewidth=2)
        ax3.xaxis.set_major_locator(MultipleLocator(2))  # per 2 epoches
        ax3.set_xlabel('Epoches', fontsize=16)
        ax3.set_ylabel('PCC on test set', fontsize=16)
        ax3.legend(loc='upper right')

        fig1.savefig('%s/Epoch_test_metrics.png' %
                     self.train_result_path, dpi=300, bbox_inches='tight')
        plt.close("all")
        plt.pause(2)
