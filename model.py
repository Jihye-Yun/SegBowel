import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from tensorboardX import SummaryWriter

from data import BowelDataset, AverageValue
from layers import inconv, down, up, outconv


class SegBowel(object):

    def __init__(self, args):
        print ("<<< SegBowel >>>")
        
        self.gpu_id = args.gpu_id

        # train configuration
        self.data_train = args.data_train
        self.data_valid = args.data_valid
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.decay = args.decay
        self.logdir = args.logdir

        # test configuration
        self.weights = args.weights
        self.data_test = args.data_test

        # build model
        self.model = Unet()
        self.criteria = nn.BCELoss()
        self.accuracy = self.accuracy4seg

        torch.cuda.set_device(self.gpu_id)
        torch.cuda.manual_seed(0)
        self.model.cuda()

        summary(self.model, (1, 448, 448))


    def train(self):
        print (f"- mode: train {self.epochs} epochs")

        # optimizer
        print (f"- optimizer: Adam (lr={self.lr}, decay={self.decay})")
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

        # summary writer
        summary_writer = SummaryWriter(os.path.join(self.logdir, "tensorboard"))

        # data_loader
        print ("- training:")
        train_dataset = BowelDataset(self.data_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        print (f"  => {len(train_dataset)} images")

        print ("- validation:")
        valid_dataset = BowelDataset(self.data_valid)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        print (f"  => {len(valid_dataset)} images")

        # training
        train_loss, train_acc = AverageValue(), AverageValue()
        valid_loss, valid_acc = AverageValue(), AverageValue()  
        for epoch in range(self.epochs):
            ##### train -->
            self.model.train()
            for i, (image, label) in enumerate(train_loader):
                image, label = image.cuda(), label.cuda()
                pred_label = self.model(image)

                cur_loss = self.criteria(pred_label, label)
                cur_acc = self.accuracy(pred_label, label)

                train_loss.update(cur_loss.item(), image.size(0))
                train_acc.update(cur_acc.item(), image.size(0))

                info = f"Epoch [{epoch+1}/{self.epochs}] ({i+1}/{len(train_loader)}): train_loss = {train_loss.val:.4f} (avg: {train_loss.avg:.4f}) train_acc = {train_acc.val:.4f} (avg: {train_acc.avg:.4f})"
                print (info, end='\r')

                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()            
                #break # batch
            #### <- train

            print (info)

            save_file = os.path.join(self.logdir, f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save({'epoch': epoch+1, 'state_dict': self.model.state_dict(), 'optimizer': optimizer.state_dict()}, save_file)

            summary_writer.add_scalar('train/loss', train_loss.avg, epoch)
            summary_writer.add_scalar('train/acc', train_acc.avg, epoch)

            ##### valid -->
            self.model.eval()
            with torch.no_grad():
                for i, (image, label) in enumerate(valid_loader):
                    image, label = image.cuda(), label.cuda()
                    pred_label = self.model(image)

                    cur_loss = self.criteria(pred_label, label)
                    cur_acc = self.accuracy(pred_label, label)

                    valid_loss.update(cur_loss.item(), image.size(0))
                    valid_acc.update(cur_acc.item(), image.size(0))

                    info = f"\t\t({i+1}/{len(valid_loader)}): valid_loss = {valid_loss.val:.4f} (avg: {valid_loss.avg:.4f}) valid_acc = {valid_acc.val:.4f} (avg: {valid_acc.avg:.4f})"
                    print (info, end='\r')
                    #break # batch
            ##### <- valid

            print (info)

            summary_writer.add_scalar('valid/loss', valid_loss.avg, epoch)
            summary_writer.add_scalar('valid/acc', valid_acc.avg, epoch)            
            
            #break # epoch

        summary_writer.close()


    def accuracy4seg(self, pred, mask):
        TH = 0.5

        pred = pred.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()

        res = np.equal(np.where(pred > TH, 1, 0), mask)
        res = np.sum(res) / len(pred.flatten())

        return res


    def test(self):
        print (f"- mode: test (trained weight: {self.weights}")
        self.model.load_state_dict(torch.load(self.weights, map_location=('cuda:'+str(self.gpu_id)))['state_dict'])
        
        print ("- test:")
        test_dataset = BowelDataset(self.data_test, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        print (f"  => {len(test_dataset)} images")

        
        #### test -->
        self.model.eval()
        with torch.no_grad():
            test_results = []
            #for i, (image, label) in enumerate(test_loader):
            for i, image in enumerate(test_loader):
                #image, label = image.cuda(), label.cuda()
                image = image.cuda()
                pred_label = self.model(image)
                test_results.extend(np.squeeze(pred_label.cpu().detach().numpy(), 1))
                #break # batch

            test_results = np.array(test_results)
            print (test_results.shape)
        #### <- test

        # save results
        test_dataset.save_results(test_results)


class Unet(nn.Module):
    
    def __init__(self, n_channels=1, n_classes=1, downsize_nb_filters_factor=2):
        super(Unet, self).__init__()

        self.inc = inconv(n_channels, 64//downsize_nb_filters_factor)
        self.down1 = down(64//downsize_nb_filters_factor, 128//downsize_nb_filters_factor)
        self.down2 = down(128//downsize_nb_filters_factor, 256//downsize_nb_filters_factor)
        self.down3 = down(256//downsize_nb_filters_factor, 512//downsize_nb_filters_factor)
        self.down4 = down(512//downsize_nb_filters_factor, 512//downsize_nb_filters_factor)
        self.up1 = up(1024//downsize_nb_filters_factor, 256//downsize_nb_filters_factor)
        self.up2 = up(512//downsize_nb_filters_factor, 128//downsize_nb_filters_factor)
        self.up3 = up(256//downsize_nb_filters_factor, 64//downsize_nb_filters_factor)
        self.up4 = up(128//downsize_nb_filters_factor, 64//downsize_nb_filters_factor)
        self.outc = outconv(64//downsize_nb_filters_factor, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)