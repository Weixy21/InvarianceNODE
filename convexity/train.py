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
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
exp_id = '0'
t = torch.linspace(0., 10., args.data_size).to(device)  # x
off_t = torch.linspace(2., 0.1, args.data_size).to(device)
y = t + off_t
z = t + off_t/2

fx = (t**2).unsqueeze(1)
fy = (y**2).unsqueeze(1)
fz = (z**2).unsqueeze(1)
fval = torch.cat([fx, fy, fz], dim = 1)
fval = fval.unsqueeze(1).to(device)
fval0 = fval[0,:,:]


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
    batch_fval0 = fval[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_fval = torch.stack([fval[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_fval0.to(device), batch_t.to(device), batch_fval.to(device)


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
        ax_traj.set_ylabel('x, y, z')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], 'r-', t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-', t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 2], 'b-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], 'r--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'g--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 2], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Convex Portrait')
        ax_phase.set_xlabel('t')
        ax_phase.set_ylabel('b(x)')
        ax_phase.plot(t.cpu().numpy(), 0.5*(true_y.cpu().numpy()[:, 0, 0] + true_y.cpu().numpy()[:, 0, 1]) - true_y.cpu().numpy()[:, 0, 2], 'g-')
        ax_phase.plot(t.cpu().numpy(), 0.5*(pred_y.cpu().numpy()[:, 0, 0] + pred_y.cpu().numpy()[:, 0, 1]) - pred_y.cpu().numpy()[:, 0, 2], 'b--')
        ax_phase.plot(t.cpu().numpy(), torch.zeros_like(t).cpu().numpy(), 'k--')

        fig.tight_layout()

        plt.savefig('logs/png' + exp_id +'/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 3),
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        self.use_dcbf = False

    def forward(self, t, y):
        return self.net(y)
    

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
    
    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    # tt = torch.tensor([0., 0.1]).to(device)
    for itr in range(1, args.niters + 1):
        func.use_dcbf = False
        func.train()
        optimizer.zero_grad()
        batch_fval0, batch_t, batch_fval = get_batch()
        pred_fval = odeint(func, batch_fval0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_fval - batch_fval))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())


        if itr % args.test_freq == 0:
            with torch.no_grad():

                pred_fval = odeint(func, fval0, t)
                loss = torch.mean(torch.abs(pred_fval - fval))

                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(fval, pred_fval, ii)
                ii += 1

            torch.save(func.state_dict(), "./logs/model"+ exp_id + "/model_node_itr" + format(itr, '02d') + ".pth")
            
        end = time.time()

    