
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import itertools


from model.dgmodel import img2state ,state2img ,imgDmodel ,stateDmodel ,state2state
from data.gymdata import Robotdata
from utils.utils import ImagePool ,GANLoss
from model.fmodel import Fmodel ,ImgFmodel ,ADmodel ,AGmodel


class SSOriginCycleGANModel():
    def __init__(self ,opt):
        self.opt = opt
        self.isTrain = opt.istrain
        self.Tensor = torch.cuda.FloatTensor

        self.netG_A = state2state(opt=self.opt).cuda()
        self.netG_B = state2state(opt=self.opt).cuda()
        self.net_action_G_A = AGmodel(flag='A2B' ,opt=self.opt).cuda()
        self.net_action_G_B = AGmodel(flag='B2A' ,opt=self.opt).cuda()
        self.netF_A = Fmodel(self.opt).cuda()
        self.netF_B = ImgFmodel(opt=self.opt).cuda()
        self.dataF = Robotdata.get_loader(opt)
        self.train_forward_state(pretrained=opt.pretrain_f)
        # self.train_forward_img(pretrained=True)

        self.reset_buffer()


        # if self.isTrain:
        self.netD_A = stateDmodel(opt=self.opt).cuda()
        self.netD_B = stateDmodel(opt=self.opt).cuda()
        self.net_action_D_A = ADmodel(opt=self.opt).cuda()
        self.net_action_D_B = ADmodel(opt=self.opt).cuda()

        # if self.isTrain:
        self.fake_A_pool = ImagePool(pool_size=128)
        self.fake_B_pool = ImagePool(pool_size=128)
        self.fake_action_A_pool = ImagePool(pool_size=128)
        self.fake_action_B_pool = ImagePool(pool_size=128)
        # define loss functions
        self.criterionGAN = GANLoss(tensor=self.Tensor).cuda()
        if opt.loss == 'l1':
            self.criterionCycle = nn.L1Loss()
        elif opt.loss == 'l2':
            self.criterionCycle = nn.MSELoss()
        self.ImgcriterionCycle = nn.MSELoss()
        self.StatecriterionCycle = nn.L1Loss()
        # initialize optimizers
        parameters = [{'params' :self.netF_A.parameters() ,'lr' :self.opt.F_lr},
                      {'params': self.netF_B.parameters(), 'lr': self.opt.F_lr},
                      {'params': self.netG_A.parameters(), 'lr': self.opt.G_lr},
                      {'params' :self.netG_B.parameters() ,'lr' :self.opt.G_lr},
                      {'params': self.net_action_G_A.parameters(), 'lr': self.opt.A_lr},
                      {'params': self.net_action_G_B.parameters(), 'lr': self.opt.A_lr}]
        self.optimizer_G = torch.optim.Adam(parameters)
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters())
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters())
        self.optimizer_action_D_A = torch.optim.Adam(self.net_action_D_A.parameters())
        self.optimizer_action_D_B = torch.optim.Adam(self.net_action_D_B.parameters())

        print('---------- Networks initialized ---------------')
        print('-----------------------------------------------')


    def train_forward_state(self ,pretrained=False):
        weight_path = os.path.join(self.opt.data_root ,'data_{}/pred.pth'.format(self.opt.test_id1))
        if pretrained:
            self.netF_A.load_state_dict(torch.load(weight_path))
            print('forward model has loaded!')
            return None
        optimizer = torch.optim.Adam(self.netF_A.parameters() ,lr=1e-3)
        loss_fn = nn.L1Loss()
        for epoch in range(50):
            epoch_loss = 0
            for i ,item in enumerate(tqdm(self.dataF)):
                state, action, result = item[1]
                state = state.float().cuda()
                action = action.float().cuda()
                result = result.float().cuda()
                out = self.netF_A(state, action)
                loss = loss_fn(out, result)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print('epoch:{} loss:{:.7f}'.format(epoch ,epoch_loss /len(self.dataF)))
            torch.save(self.netF_A.state_dict(), weight_path)
        print('forward model has been trained!')

    def train_forward_img(self ,pretrained=False):
        weight_path = './model/imgpred.pth'
        if pretrained:
            self.netF_B.load_state_dict(torch.load(weight_path))
            return None
        optimizer = torch.optim.Adam(self.netF_B.parameters() ,lr=1e-3)
        loss_fn = nn.MSELoss()
        for epoch in range(50):
            epoch_loss = 0
            for i ,item in enumerate(tqdm(self.dataF)):
                state, action, result = item[1]
                state = state.float().cuda()
                action = action.float().cuda()
                result = result.float().cuda()
                out = self.netF_B(state, action ) *100
                loss = loss_fn(out, result)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print('epoch:{} loss:{:.7f}'.format(epoch ,epoch_loss /len(self.dataF)))
            torch.save(self.netF_B.state_dict(), weight_path)
        print('forward model has been trained!')

    def set_input(self, input):
        # AtoB = self.opt.which_direction == 'AtoB'
        # input_A = input['A' if AtoB else 'B']
        # input_B = input['B' if AtoB else 'A']
        # A is state
        self.input_A = input[1][0]

        # B is img
        self.input_Bt0 = input[2][0]
        self.input_Bt1 = input[2][1]
        self.action = input[0][1]
        self.gt0 = input[2][0].float().cuda()
        self.gt1 = input[2][1].float().cuda()


    def forward(self):
        self.real_A = Variable(self.input_A).float().cuda()
        self.real_Bt0 = Variable(self.input_Bt0).float().cuda()
        self.real_Bt1 = Variable(self.input_Bt1).float().cuda()
        self.action = Variable(self.action).float().cuda()


    def test(self):
        # forward
        self.forward()
        # G_A and G_B
        self.backward_G()
        self.backward_D_B()


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        if self.isTrain:
            loss_D.backward()
        return loss_D

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_At0)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.item()

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_Bt0)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_Bt0, fake_B)
        self.loss_D_A = loss_D_A.item()


    def backward_G(self):
        lambda_G_B0 = self.opt.lambda_G0
        lambda_G_B1 = self.opt.lambda_G1
        lambda_F = self.opt.lambda_F
        lambda_C = 100.


        # GAN loss D_B(G_B(B))
        fake_At0 = self.netG_B(self.real_Bt0)
        pred_fake = self.netD_B(fake_At0)
        loss_G_Bt0 = self.criterionGAN(pred_fake, True) * lambda_G_B0

        rec_At0 = self.netG_A(fake_At0)
        loss_cycle_original_A = self.criterionCycle(rec_At0, self.real_Bt0) * lambda_C

        # GAN loss D_B(G_B(B))
        fake_Bt0 = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_Bt0)
        loss_G_At0 = self.criterionGAN(pred_fake, True) * lambda_G_B0

        rec_B = self.netG_B(fake_Bt0)
        loss_cycle_original_B = self.criterionCycle(rec_B, self.real_A) * lambda_C




        fake_At1 = self.netF_A(fake_At0 ,self.action)
        pred_At1 = self.netG_B(self.real_Bt1)
        self.loss_state_lt0 = self.criterionCycle(fake_At0, self.gt0)
        self.loss_state_lt1 = self.criterionCycle(pred_At1, self.gt1)

        # combined loss
        loss_G = loss_G_Bt0 + loss_cycle_original_A + loss_G_At0 + loss_cycle_original_B

        if self.isTrain:
            loss_G.backward()

        self.fake_At0 = fake_At0.data
        self.fake_Bt0 = fake_Bt0.data
        self.fake_At1 = fake_At1.data

        self.loss_G_Bt0 = loss_G_Bt0.item()
        self.loss_G_At0 = loss_G_At0.item()
        self.loss_cycle_original = loss_cycle_original_A.item()+loss_cycle_original_B.item()
        # self.loss_G_Bt1 = loss_G_Bt1.item()
        # self.loss_cycle = loss_cycle.item()

        self.loss_state_lt0 = self.loss_state_lt0.item()
        self.loss_state_lt1 = self.loss_state_lt1.item()
        self.gt_buffer0.append(self.gt0.cpu().data.numpy())
        self.pred_buffer0.append(self.fake_At0.cpu().data.numpy())
        self.gt_buffer1.append(self.gt1.cpu().data.numpy())
        self.pred_buffer1.append(self.fake_At1.cpu().data.numpy())

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('L_t0' ,self.loss_state_lt0), ('L_t1' ,self.loss_state_lt1),
                                  ('D_B', self.loss_D_B), ('G_B0', self.loss_G_Bt0),
                                  ('G_A0', self.loss_G_At0), ('Cyc',  self.loss_cycle_original)])
        return ret_errors

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, path):
        save_filename = 'model_{}.pth'.format(network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    def save(self, path):
        self.save_network(self.netG_B, 'G_B', path)
        self.save_network(self.netD_B, 'D_B', path)
        self.save_network(self.netG_A, 'G_A', path)
        self.save_network(self.netD_A, 'D_A', path)

        self.save_network(self.net_action_G_B, 'action_G_B', path)
        self.save_network(self.net_action_D_B, 'action_D_B', path)
        self.save_network(self.net_action_G_A, 'action_G_A', path)
        self.save_network(self.net_action_D_A, 'action_D_A', path)

    def load_network(self, network, network_label, path):
        weight_filename = 'model_{}.pth'.format(network_label)
        weight_path = os.path.join(path, weight_filename)
        network.load_state_dict(torch.load(weight_path))

    def load(self ,path):
        self.load_network(self.netG_B, 'G_B', path)
        self.load_network(self.netD_B, 'D_B', path)
        self.load_network(self.netG_A, 'G_A', path)
        self.load_network(self.netD_A, 'D_A', path)

        self.load_network(self.net_action_G_B, 'action_G_B', path)
        self.load_network(self.net_action_D_B, 'action_D_B', path)
        self.load_network(self.net_action_G_A, 'action_G_A', path)
        self.load_network(self.net_action_D_A, 'action_D_A', path)

    def show_points(self ,gt_data ,pred_data):
        ncols = int(np.sqrt(gt_data.shape[1]))
        nrows = int(np.sqrt(gt_data.shape[1]) ) +1
        assert (ncols *nrows>=gt_data.shape[1])
        _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()

        for ax_i, ax in enumerate(axes):
            if ax_i>=gt_data.shape[1]:
                continue
            ax.scatter(gt_data[:, ax_i], pred_data[:, ax_i], s=3, label='xyz_{}'.format(ax_i))


    def npdata(self ,item):
        return item.cpu().data.numpy()

    def reset_buffer(self):
        self.gt_buffer0 = []
        self.pred_buffer0 = []
        self.gt_buffer1 = []
        self.pred_buffer1 = []


    def visual(self ,path):
        gt_data = np.vstack(self.gt_buffer0)
        pred_data = np.vstack(self.pred_buffer0)
        self.show_points(gt_data ,pred_data)
        plt.legend()
        plt.savefig(path)
        plt.cla()
        plt.clf()

        gt_data = np.vstack(self.gt_buffer1)
        pred_data = np.vstack(self.pred_buffer1)
        self.show_points(gt_data, pred_data)
        plt.legend()
        plt.savefig(path.replace('.jpg' ,'_step1.jpg'))
        self.reset_buffer()

