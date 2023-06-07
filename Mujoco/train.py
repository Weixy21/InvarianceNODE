
import os
import argparse
import time
import numpy as np
import copy

import torch
import torch.nn as nn
from cvxopt import solvers, matrix
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

import walker2d_data #import load_dataset
import cheetah_data

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--seq_len', type=int, default=20)
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument("--dataset", default="walker2d")
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--consider_noise', action='store_true')
parser.add_argument('--hard_truncate', action='store_true')
parser.add_argument('--num_var', type=int, default=1)
parser.add_argument('--use_dcbf', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def cvx_solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    return sol['x']

if args.dataset == "walker2d":
    trainloader, testloader, in_features = walker2d_data.load_dataset(
        seq_len=args.seq_len
    )
    low = torch.from_numpy(
        np.array(
            [
                0.8306609392166138,
                -0.629609227180481,
                -1.1450262069702148,
                -1.2276967763900757,
                -1.2756223678588867,
                -0.9984434843063354,
                -1.2026487588882446,
                -1.4123592376708984,
                -2.2933568954467773,
                -3.8838634490966797,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
            ]
        )
    ).to(device)

    high = torch.from_numpy(
        np.array(
            [
                1.6924704313278198,
                0.9378023743629456,
                0.1535143405199051,
                0.16713030636310577,
                1.2674068212509155,
                0.2375510334968567,
                0.18477311730384827,
                1.2249037027359009,
                5.372719764709473,
                3.470100164413452,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
            ]
        )
    ).to(device)
elif args.dataset == "halfcheetah":
    trainloader, testloader, in_features = cheetah_data.load_dataset(
        seq_len=args.seq_len
    )
    low = torch.from_numpy(
        np.array(
            [
                -0.2936522662639618,
                -0.29987457394599915,
                -0.5890289545059204,
                -0.7020226120948792,
                -0.518144965171814,
                -0.7397885918617249,
                -0.841916024684906,
                -0.6131184697151184,
                -2.262007713317871,
                -2.206634759902954,
                -4.519010066986084,
                -17.817516326904297,
                -17.485055923461914,
                -21.776256561279297,
                -18.28421401977539,
                -20.88571548461914,
                -19.104284286499023,
            ]
        )
    ).to(device)

    high = torch.from_numpy(
        np.array(
            [
                0.1298145353794098,
                0.4833938181400299,
                0.6584464311599731,
                0.6455742120742798,
                0.8780248761177063,
                0.7083053588867188,
                0.7404807209968567,
                0.5502343773841858,
                4.839046955108643,
                2.9650542736053467,
                4.737936019897461,
                16.746498107910156,
                22.432693481445312,
                18.447025299072266,
                22.226903915405273,
                22.869461059570312,
                15.584585189819336,
            ]
        )
    ).to(device)
else:
    raise ValueError(f"Unknown dataset '{args.dataset}'")

class ODEFunc(nn.Module):

    def __init__(self,dim):
        super(ODEFunc, self).__init__()

        latent_dim = 64
        self.use_dcbf = False
        self.study_obs = False
        self.pos_x = 0
        self.knee_left = 0
        self.knee_right = 0
        self.hip_joint = 0
        self.net = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        if self.use_dcbf:
            w_ref = self.net[4].weight[:,0:args.num_var].clone().detach()
            new_w = self.dCBF(y, w_ref)
            temp_net = copy.deepcopy(self.net)
            temp_net[4].weight[:,0:args.num_var] = new_w            
            out = temp_net(y)
            return out
        else:
            return self.net(y)
    
    def dCBF(self, y, w_ref):
        # get the coe of controls
        if self.training and self.use_dcbf:
            y = y.view(args.batch_time*args.batch_size, -1)
        nBatch = 1#y.shape[0]
        u_rt = []
        for i in range(args.num_var):
            temp_net = copy.deepcopy(self.net)
            for param in temp_net.parameters():
                param.requires_grad = False
            if i > 0:
                temp_net[4].weight[:,0:i] = torch.zeros_like(temp_net[4].weight[:,0:i])
            temp_net[4].weight[:,i] = torch.ones_like(temp_net[4].weight[:,i])
            temp_net[4].weight[:,i+1:] = torch.zeros_like(temp_net[4].weight[:,i+1:])
            temp_net[4].bias[:] = torch.zeros_like(temp_net[4].bias[:])
            u_rt.append(temp_net(y).detach())
        
        # get the remaining values
        temp_net =copy.deepcopy(self.net)
        for param in temp_net.parameters():
            param.requires_grad = False
        for i in range(args.num_var):
            temp_net[4].weight[:,i] = torch.zeros_like(temp_net[4].weight[:,i])

        bias_rt = temp_net(y).detach()
        G, h = [], []

        if self.study_obs:
            ############################################ceiling
            # if self.pos_x >= 1.5:
            #     height = 1.6
            # else:
            #     height = 2
            # b = height - y[0] - 0.3*y[9]  # 1.69
            # Lfb = -bias_rt[0] - 0.3*bias_rt[9] 
            # Lgbu = []
            # for k in range(args.num_var):
            #     lgbk = torch.zeros(nBatch, 17)
            #     lgbk[0,0] = -u_rt[k][0] 
            #     lgbk[0,9] = -0.3*u_rt[k][9]
            #     Lgbu.append(-lgbk)
            # G = torch.cat(Lgbu, dim = 1)
            # G = torch.reshape(G, (nBatch, 1, args.num_var*17)).to(device)
            # h = (torch.reshape((Lfb + 10*b), (nBatch, 1))).to(device) 

            #############################################ball
            b = (self.pos_x - 2)**2 + (y[0]- 1.8)**2 - 0.6**2
            Lfb = 2*(self.pos_x - 2)*y[8]*0.01 + 2*(y[0]- 1.8)*bias_rt[0] 
            Lgbu = []
            for k in range(args.num_var):
                lgbk = torch.zeros(nBatch, 17)
                # lgbk[0,8] = 2*(self.pos_x - 1.5)*u_rt[k][8]*0.01 
                lgbk[0,0] = 2*(y[0]- 1.8)*u_rt[k][0]
                Lgbu.append(-lgbk)
            G = torch.cat(Lgbu, dim = 1)
            G = torch.reshape(G, (nBatch, 1, args.num_var*17)).to(device)
            h = (torch.reshape((Lfb + 20*b), (nBatch, 1))).to(device) 


            b = (self.pos_x - 3)**2 + (y[0]- 1.8)**2 - 0.6**2
            Lfb = 2*(self.pos_x - 3)*y[8]*0.01 + 2*(y[0]- 1.8)*bias_rt[0] 
            Lgbu = []
            for k in range(args.num_var):
                lgbk = torch.zeros(nBatch, 17)
                # lgbk[0,8] = 2*(self.pos_x - 1.5)*u_rt[k][8]*0.01 
                lgbk[0,0] = 2*(y[0]- 1.8)*u_rt[k][0]
                Lgbu.append(-lgbk)
            G1 = torch.cat(Lgbu, dim = 1)
            G1 = torch.reshape(G1, (nBatch, 1, args.num_var*17)).to(device)
            h1 = (torch.reshape((Lfb + 20*b), (nBatch, 1))).to(device) 

            b = (self.pos_x - 4)**2 + (y[0]- 1.8)**2 - 0.6**2
            Lfb = 2*(self.pos_x - 4)*y[8]*0.01 + 2*(y[0]- 1.8)*bias_rt[0] 
            Lgbu = []
            for k in range(args.num_var):
                lgbk = torch.zeros(nBatch, 17)
                # lgbk[0,8] = 2*(self.pos_x - 1.5)*u_rt[k][8]*0.01 
                lgbk[0,0] = 2*(y[0]- 1.8)*u_rt[k][0]
                Lgbu.append(-lgbk)
            G2 = torch.cat(Lgbu, dim = 1)
            G2 = torch.reshape(G2, (nBatch, 1, args.num_var*17)).to(device)
            h2 = (torch.reshape((Lfb + 20*b), (nBatch, 1))).to(device) 

            G = torch.cat([G, G1, G2], dim = 1)
            h = torch.cat([h, h1, h2], dim = 1)
        else:
        ###############################################joint limitations
            for i in range(17):
                b = y[i] - low[i]
                Lfb = bias_rt[i]
                Lgbu = []
                for k in range(args.num_var):
                    lgbk = torch.zeros(nBatch, 17)
                    lgbk[0,i] = u_rt[k][i]
                    Lgbu.append(-lgbk)
                G1 = torch.cat(Lgbu, dim = 1)
                G1 = torch.reshape(G1, (nBatch, 1, args.num_var*17)).to(device)
                h1 = (torch.reshape((Lfb + 10*b), (nBatch, 1))).to(device)  # coe ??

                G.append(G1)
                h.append(h1)
                
                b = high[i] - y[i]
                Lfb = -bias_rt[i]
                Lgbu = []
                for k in range(args.num_var):
                    lgbk = torch.zeros(nBatch, 17)
                    lgbk[0,i] = -u_rt[k][i]
                    Lgbu.append(-lgbk)
                G2 = torch.cat(Lgbu, dim = 1)
                G2 = torch.reshape(G2, (nBatch, 1, args.num_var*17)).to(device)
                h2 = (torch.reshape((Lfb + 10*b), (nBatch, 1))).to(device)  # coe ??

                G.append(G2)
                h.append(h2)
            G = torch.cat(G, dim = 1).to(device)
            h = torch.cat(h, dim = 1).to(device)

        q = -w_ref.transpose(1,0).flatten().expand(nBatch, args.num_var*17).to(device)  # need to check
        Q = Variable(torch.eye(args.num_var*17))
        Q = Q.unsqueeze(0).expand(nBatch, args.num_var*17, args.num_var*17).to(device)  #could also add some trainable parameters in Q
        
        x = cvx_solver(Q[0].detach().double(), q[0].detach().double(), G[0].detach().double(), h[0].detach().double())
        out = []
        for i in range(17*args.num_var):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(device)
        out = out.view(args.num_var, 17).transpose(1,0)  # need to check
        
        return out

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

if __name__ == '__main__':

    ii = 0

    os.makedirs("ckpt",exist_ok=True)
    #trainloader, testloader, in_features = load_dataset(seq_len=args.seq_len)

    func = ODEFunc(in_features).to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    loss_meter = RunningAverageMeter()

    for ep in range(args.epochs):
        pbar = tqdm(total=len(trainloader))
        for batch_y0, batch_y in iter(trainloader):
            optimizer.zero_grad()
            batch_y0 = batch_y0.to(device)
            batch_t = torch.linspace(0,args.seq_len,args.seq_len).to(device)
            batch_y = batch_y.to(device)

            pred_y = odeint(func, batch_y0, batch_t)
            if args.hard_truncate:
                trunc_y = torch.clip(pred_y, low, high)
            else:
                trunc_y = pred_y

            loss = torch.mean(torch.abs(trunc_y - batch_y))
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.detach().cpu().item())
            pbar.update(1)
            pbar.set_description(f"train_loss={loss_meter.val:0.4g}")
        scheduler.step()
        pbar.close()
        with torch.no_grad():
            test_losses = []
            for batch_y0, batch_y in iter(testloader):
                batch_y0 = batch_y0.to(device)
                batch_t = torch.linspace(0, args.seq_len, args.seq_len).to(device)
                batch_y = batch_y.to(device)
                pred_y = odeint(func, batch_y0, batch_t)
                if args.hard_truncate:
                    trunc_y = torch.clip(pred_y, low, high)
                else:
                    trunc_y = pred_y
                loss = torch.mean(torch.abs(trunc_y - batch_y)).detach().cpu().item()
                test_losses.append(loss)
            print(f"Epoch {ep+1} test_loss={np.mean(test_losses):0.4g}")
        if args.hard_truncate:
            torch.save(func.state_dict(), f"ckpt_tunc/walker2d_ep{ep+1:03d}.pth")
        else:
            torch.save(func.state_dict(), f"ckpt/walker2d_ep{ep+1:03d}.pth")
