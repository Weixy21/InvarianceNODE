import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from cvxopt import solvers, matrix
import copy
from qpth.qp import QPFunction, QPSolvers
import pickle
from torch.nn.functional import normalize

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--consider_noise', action='store_true')
parser.add_argument('--num_var', type=int, default=1)
parser.add_argument('--use_cnn', action='store_true')
parser.add_argument('--use_fc', action='store_true')
parser.add_argument('--use_diff_weight', action='store_true')

args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
exp_id = '0'  
t = torch.linspace(0., 9.9, args.data_size).to(device)


pkl_file = open('data3.pkl','rb')
data = pickle.load(pkl_file)
lidar = torch.tensor(data['lidar']).float()  # batch, seq, data
control = torch.tensor(data['ctrl']).float()
ego = torch.tensor(data['ego']).float()
other = torch.tensor(data['other']).float()
pkl_file.close()

lidar_bat = lidar.unsqueeze(2).to(device)   # batch, seq, data_h x data_w
control_bat = control.unsqueeze(2).to(device)
ego_bat = ego.unsqueeze(2).to(device)
other_bat = other.unsqueeze(2).to(device)

#test data
lidar = lidar_bat[-1,:,:,:]
control = control_bat[-1,:,:,:]
ego = ego_bat[-1,:,:,:]
other = other_bat[-1,:,:,:]

control0 = control_bat[-1,0,:,:]
lidar0 = lidar_bat[-1,0,:,:]
ego0 = ego_bat[-1,0,:,:]



def cvx_solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    return sol['x']


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_u0 = control[s]  # (M, D)
    batch_lidar0 = lidar[s]
    batch_t = t[:args.batch_time]  # (T)
    batch_u = torch.stack([control[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    batch_lidar = torch.stack([lidar[s + i] for i in range(args.batch_time)], dim=0)
    return batch_u0.to(device), batch_lidar0.to(device), batch_t.to(device), batch_u.to(device), batch_lidar.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('logs/png' + exp_id)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4), facecolor='white')
    ax_traj = fig.add_subplot(121, frameon=False)
    ax_phase = fig.add_subplot(122, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('u1,u2')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('u1')
        ax_phase.set_ylabel('u2')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')

        fig.tight_layout()

        plt.savefig('logs/png' + exp_id +'/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self, fc_param, conv_filters,
                 dropout= 0.2):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(fc_param[0], fc_param[1]),
            nn.GELU(), #silu(), gelu()
            nn.Linear(fc_param[1], fc_param[2]),
            nn.GELU(),  #Tanh
            nn.Linear(fc_param[2], fc_param[3])  #drift and affine terms
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        self.use_dcbf = False
        self.lidar = lidar0
        # self.pa = Parameter(torch.ones(1))
        # self.pb = Parameter(torch.ones(1))
        if args.use_cnn == True:
            self._cnn = self._build_cnn(conv_filters, dropout=dropout)
        if args.use_fc == True:
            self._fc = nn.Sequential(
                nn.Linear(100, 200),
                nn.Tanh(),
                nn.Linear(200, 12)
            )

    def forward(self, t, y):
        if self.training:
            if args.use_fc == True:
                sensor = self._fc(self.lidar)
            g = self.net(y)
            u1, u2 = torch.chunk(g[:,:,2:], 2, dim=-1)
            if args.use_fc == True:
                out = torch.cat([(u1*sensor).sum(axis = 2).unsqueeze(2), (u2*sensor).sum(axis = 2).unsqueeze(2)], dim = 2) + g[:,:,0:2]
            else:
                out = torch.cat([(u1*self.lidar).sum(axis = 2).unsqueeze(2), (u2*self.lidar).sum(axis = 2).unsqueeze(2)], dim = 2) + g[:,:,0:2] #affine form
        else:
            if args.use_fc == True:
                sensor = self._fc(self.lidar)
            g = self.net(y)
            u1, u2 = torch.chunk(g[:,2:], 2, dim=-1)
            if args.use_fc == True:
                out = torch.cat([(u1*sensor).sum(axis = 1).unsqueeze(1), (u2*sensor).sum(axis = 1).unsqueeze(1)], dim = 1) + g[:,0:2]
            else:
                out = torch.cat([(u1*self.lidar).sum(axis = 1).unsqueeze(1), (u2*self.lidar).sum(axis = 1).unsqueeze(1)], dim = 1) + g[:,0:2]
        return out
    
    def _build_cnn(self, filters, dropout=0., no_act_last_layer=False):
        
        modules = nn.ModuleList()
        for i, filt in enumerate(filters):
            modules.append(nn.Conv1d(*filt))
            if (i != len(filters) - 1) or (not no_act_last_layer):
                modules.append(nn.BatchNorm1d(filt[1]))
                modules.append(nn.ReLU())
                if dropout > 0:
                    modules.append(nn.Dropout(p=dropout))
        modules = nn.Sequential(*modules)
        
        return modules


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
    makedirs('logs/model' + exp_id)
    
    if args.use_cnn or args.use_fc:
        fc_param = [2, 64, 128, 26]  # [2, 32, 64, 26] 
    else:
        fc_param = [2, 50, 400, 202]
    #channel x output channels x kernel size x stride x padding
    conv_param = [[1, 4, 5, 2, 1], [4, 8, 3, 2, 1], [8, 12, 3, 2, 0]]
    func = ODEFunc(fc_param, conv_param).to(device)
    func.lidar = lidar0
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    tt = torch.tensor([0., 0.1]).to(device)
    for itr in range(1, args.niters + 1):
        func.use_dcbf = False
        func.train()
        optimizer.zero_grad()
        batch_u0, batch_lidar0, batch_t, batch_u, batch_lidar = get_batch()
        u0 = batch_u0
        pred_u = u0.unsqueeze(0)
        for i in range(args.batch_time-1):
            # func.lidar = normalize(batch_lidar[i,:,:,:], p = 2, dim = 2)
            sensor = batch_lidar[i,:,:,:]/30.
            if args.use_cnn == True:
                sensor = func._cnn(sensor)
                sensor = sensor.max(dim=-1)[0]
                func.lidar = sensor.unsqueeze(1)
            else:
                func.lidar = sensor
            pred = odeint(func, u0, tt).to(device)
            pred_u = torch.cat([pred_u, pred[-1,:,:,:].unsqueeze(0)], dim = 0)
            u0 = pred[-1,:,:,:]
        
        if args.use_diff_weight:
            loss = 0.01*torch.mean(torch.abs(pred_u[:,:,:,0] - batch_u[:,:,:,0])) + 0.99*torch.mean(torch.abs(pred_u[:,:,:,1] - batch_u[:,:,:,1]))
        else:
            loss = torch.mean(torch.abs(pred_u - batch_u))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())


        if itr % args.test_freq == 0:
            with torch.no_grad():
                func.use_dcbf = False
                u0 = control0
                pred_u = u0.unsqueeze(0)
                func.eval()
                for i in range(args.data_size-1):
                    # func.lidar = normalize(lidar[i,:,:], p = 2, dim = 1)
                    sensor = lidar[i,:,:]/30.
                    if args.use_cnn == True:
                        x = sensor.unsqueeze(0)
                        sensor = func._cnn(x)
                        func.lidar = sensor.max(dim=-1)[0]
                    else:
                        func.lidar = sensor
                    
                    pred = odeint(func, u0, tt)
                    pred_u = torch.cat([pred_u, pred[-1,:,:].unsqueeze(0)], dim = 0)
                    u0 = pred[-1,:,:]
                if args.use_diff_weight:
                    loss = 0.01*torch.mean(torch.abs(pred_u[:,:,0] - control[:,:,0])) + 0.99*torch.mean(torch.abs(pred_u[:,:,1] - control[:,:,1]))
                else:
                    loss = torch.mean(torch.abs(pred_u - control))

                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(control, pred_u, ii)
                ii += 1
            torch.save(func.state_dict(), "./logs/model"+ exp_id + "/model_node_itr" + format(ii, '02d') + ".pth")
            
        end = time.time()

    