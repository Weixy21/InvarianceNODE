import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from cvxopt import solvers, matrix
from qpth.qp import QPFunction, QPSolvers
import copy

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--num_var', type=int, default=1)
parser.add_argument('--consider_noise', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = 'cpu'

t = torch.linspace(0., 10., args.data_size).to(device)  # x, 13.
off_t = torch.linspace(2., 0.1, args.data_size).to(device)

# off_t = torch.linspace(2., 0.1, 100).to(device)
# off_t2 = torch.linspace(0.1, 0.01, 30).to(device)
# off_t = torch.cat([off_t, off_t2], dim = 0)

y = t + off_t
z = t + off_t/2

fx = (t**2).unsqueeze(1)
fy = (y**2).unsqueeze(1)
fz = (z**2).unsqueeze(1)
fval = torch.cat([fx, fy, fz], dim = 1)
fval = fval.unsqueeze(1).to(device)
fval0 = fval[0,:,:]


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('test_png')
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    fig = plt.figure(figsize=(8, 4), facecolor='white')
    # ax_traj_un = fig.add_subplot(131, frameon=False)
    ax_traj = fig.add_subplot(121, frameon=False)
    ax_phase = fig.add_subplot(122, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, pred_y2, pred_y_unsafe, itr):

    if args.viz:
        # ax_traj_un.cla()
        # ax_traj_un.set_title('(a) neural ODE')
        # ax_traj_un.set_xlabel('$t/s$')
        # ax_traj_un.set_ylabel('$g(x), g(y), g(\mu_1 x + \mu_2 y)$')
        # ax_traj_un.text(1, 80, 'solid-ground truth')
        # ax_traj_un.text(1, 70, 'dashed-prediction')
        # # ax_traj_un.text(1, 120, 'solid-ground truth')
        # # ax_traj_un.text(1, 110, 'dashed-prediction')
        # # ax_traj_un.text(1, 100, 'dotted-out of distribution area')
        # ax_traj_un.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], 'r-', t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-', t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 2], 'b-')
        # ax_traj_un.plot(t.cpu().numpy(), pred_y_unsafe.cpu().numpy()[:, 0, 0], 'r--', t.cpu().numpy(), pred_y_unsafe.cpu().numpy()[:, 0, 1], 'g--', t.cpu().numpy(), pred_y_unsafe.cpu().numpy()[:, 0, 2], 'b--')
        # # ax_traj_un.plot([10, 13,13,10,10], [100, 100, 175, 175, 100], 'k:', label = 'Out of distribution area')
        # ax_traj_un.set_xlim(t.cpu().min(), t.cpu().max())
        # # ax_traj.set_ylim(-2, 2)
        # # ax_traj_un.legend()

        ax_traj.cla()
        ax_traj.set_title('(a) Hidden invariance prediction')
        ax_traj.set_xlabel('$t/s$')
        ax_traj.set_ylabel('$g(x), g(y), g(\mu_1 x + \mu_2 y)$')
        ax_traj.text(1, 80, 'solid-ground truth')
        ax_traj.text(1, 70, 'dashed-prediction')
        # ax_traj.text(1, 120, 'solid-ground truth')
        # ax_traj.text(1, 110, 'dashed-prediction')
        # ax_traj.text(1, 100, 'dotted-out of distribution area')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], 'r-', t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-', t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 2], 'b-')
        ax_traj.plot(t.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 0], 'r--', t.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 1], 'g--', t.cpu().numpy(), pred_y2.cpu().numpy()[:, 0, 2], 'b--')
        # ax_traj.plot([10, 13,13,10,10], [100, 100, 175, 175, 100], 'k:', label = 'Out of distribution area')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        # ax_traj.set_ylim(-2, 2)
        # ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('(b) convexity portrait')
        ax_phase.set_xlabel('$t/s$')
        ax_phase.set_ylabel(r'$\mu_1 g(x) + \mu_2 g(y) - g(\mu_1 x + \mu_2 y)$')
        ax_phase.plot(t.cpu().numpy(), torch.zeros_like(t).cpu().numpy(), 'k--', label = 'convexity boundary')
        ax_phase.plot(t.cpu().numpy(), 0.5*(pred_y_unsafe.cpu().numpy()[:, 0, 0] + pred_y_unsafe.cpu().numpy()[:, 0, 1]) - pred_y_unsafe.cpu().numpy()[:, 0, 2], 'r--', label = 'neural ODE')
        ax_phase.plot(t.cpu().numpy(), 0.5*(true_y.cpu().numpy()[:, 0, 0] + true_y.cpu().numpy()[:, 0, 1]) - true_y.cpu().numpy()[:, 0, 2], 'g-', label = 'ground truth')
        ax_phase.plot(t.cpu().numpy(), 0.5*(pred_y.cpu().numpy()[:, 0, 0] + pred_y.cpu().numpy()[:, 0, 1]) - pred_y.cpu().numpy()[:, 0, 2], 'b--', label = 'output invariance')
        ax_phase.plot(t.cpu().numpy(), 0.5*(pred_y2.cpu().numpy()[:, 0, 0] + pred_y2.cpu().numpy()[:, 0, 1]) - pred_y2.cpu().numpy()[:, 0, 2], 'c--', label = 'hidden invariance')  
        # ax_phase.plot([10, 13,13,10,10], [-0.1, -0.1, 0.1, 0.1, -0.1], 'k:', label = 'Out of distribution area')
        ax_phase.legend()

        fig.tight_layout()
        plt.savefig('test_png/{:03d}.pdf'.format(itr))
        plt.draw()
        plt.pause(0.001)

def cvx_solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)

    return sol['x']


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
        self.use_dcbf = True
        self.use_hidden_layer = False
        self.saved_param = torch.zeros(args.num_var,3).to(device)

    def forward(self, t, y):
        
        if self.use_dcbf:
            if self.use_hidden_layer:
                temp_net = copy.deepcopy(self.net)
                temp_net[0].weight[0:args.num_var,:] = self.saved_param            
                out = temp_net(y)
            else:
                w_ref = self.net[2].weight[:,0:args.num_var]
                new_w = self.dCBF(y, w_ref)

                temp_net = copy.deepcopy(self.net)
                temp_net[2].weight[:,0:args.num_var] = new_w #.squeeze(0)
                
                out = temp_net(y)
        else:
            out = self.net(y)
        return out

    def dCBF_hidden(self, y, w_ref):

        if self.training and self.use_dcbf:
            y = y.view(args.batch_time*args.batch_size, -1)
        nBatch = y.shape[0]
        
        temp_net =copy.deepcopy(self.net)
        temp_net[0].weight[0:args.num_var,:] = self.saved_param

        #HOCBF for the Jensen
        out = temp_net(y)
        b = 0.5*(y[:, 0] + y[:, 1]) - y[:,2]
        Lfb = 0.5*(out[:,0] + out[:,1]) - out[:,2]
        LgLfbu, Lg_rm = [], 0
        p1, p2 = 10, 10  
        for i in range(args.num_var):
            z = nn.Tanh()(y[:,0]*temp_net[0].weight[i,0] + y[:,1]*temp_net[0].weight[i,1] + y[:,2]*temp_net[0].weight[i,2] + temp_net[0].bias[i])
            LgLfbu1 = torch.reshape(0.5*temp_net[2].weight[0,i]*(1 - z**2)*y[:,0] + 0.5*temp_net[2].weight[1,i]*(1 - z**2)*y[:,0] - temp_net[2].weight[2,i]*(1 - z**2)*y[:,0], (nBatch,1)) 
            LgLfbu2 = torch.reshape(0.5*temp_net[2].weight[0,i]*(1 - z**2)*y[:,1] + 0.5*temp_net[2].weight[1,i]*(1 - z**2)*y[:,1] - temp_net[2].weight[2,i]*(1 - z**2)*y[:,1], (nBatch,1))
            LgLfbu3 = torch.reshape(0.5*temp_net[2].weight[0,i]*(1 - z**2)*y[:,2] + 0.5*temp_net[2].weight[1,i]*(1 - z**2)*y[:,2] - temp_net[2].weight[2,i]*(1 - z**2)*y[:,2], (nBatch,1))
            LgLfbu.append(-LgLfbu1)
            LgLfbu.append(-LgLfbu2)
            LgLfbu.append(-LgLfbu3)
        for i in range(50):
            z = nn.Tanh()(y[:,0]*temp_net[0].weight[i,0] + y[:,1]*temp_net[0].weight[i,1] + y[:,2]*temp_net[0].weight[i,2] + temp_net[0].bias[i])
            Lg_rm = Lg_rm + 0.5*temp_net[2].weight[0,i]*(1 - z**2)*(temp_net[0].weight[i,0]*out[:,0] + temp_net[0].weight[i,1]*out[:,1] + temp_net[0].weight[i,2]*out[:,2]) + \
                            0.5*temp_net[2].weight[1,i]*(1 - z**2)*(temp_net[0].weight[i,0]*out[:,0] + temp_net[0].weight[i,1]*out[:,1] + temp_net[0].weight[i,2]*out[:,2]) - \
                              1*temp_net[2].weight[2,i]*(1 - z**2)*(temp_net[0].weight[i,0]*out[:,0] + temp_net[0].weight[i,1]*out[:,1] + temp_net[0].weight[i,2]*out[:,2])
        Lf2b = Lg_rm
        G = torch.cat(LgLfbu, dim = 1)
        G = torch.reshape(G, (nBatch, 1, args.num_var*3)).to(device)
        G0 = torch.zeros(nBatch, 1, args.num_var*3).to(device)
        G = torch.cat([G,G0], dim = 2)
        h = (torch.reshape((Lf2b + p1*Lfb + p2*(Lfb + p1*b)), (nBatch, 1))).to(device)
        

        #CLFs for stabilizing the param
        eps = 10
        for i in range(args.num_var):
            Ga = torch.zeros(nBatch, 1, 6*args.num_var).to(device)
            V1 = (temp_net[0].weight[i,0] - w_ref[i,0])**2
            LfV1 = 0
            LgV1 = 2*(temp_net[0].weight[i,0] - w_ref[i,0])
            Ga[0,0,3*i] = LgV1
            Ga[0,0,3*args.num_var+3*i] = -1
            ha = torch.reshape(-LfV1 - eps*V1, (nBatch, 1)).to(device)

            Gb = torch.zeros(nBatch, 1, 6*args.num_var).to(device)
            V2 = (temp_net[0].weight[i,1] - w_ref[i,1])**2
            LfV2 = 0
            LgV2 = 2*(temp_net[0].weight[i,1] - w_ref[i,1])
            Gb[0,0,3*i+1] = LgV2
            Gb[0,0,3*args.num_var+3*i+1] = -1
            hb = torch.reshape(-LfV2 - eps*V2, (nBatch, 1)).to(device)

            Gc = torch.zeros(nBatch, 1, 6*args.num_var).to(device)
            V3 = (temp_net[0].weight[i,2] - w_ref[i,2])**2
            LfV3 = 0
            LgV3 = 2*(temp_net[0].weight[i,2] - w_ref[i,2])
            Gc[0,0,3*i+2] = LgV3
            Gc[0,0,3*args.num_var+3*i+2] = -1
            hc = torch.reshape(-LfV3 - eps*V3, (nBatch, 1)).to(device)

            G = torch.cat([G, Ga, Gb, Gc], dim = 1).to(device)
            h = torch.cat([h, ha, hb, hc], dim = 1).to(device)
        
        q = torch.zeros(nBatch, args.num_var*6).to(device)
        Q = Variable(torch.eye(args.num_var*6))
        Q = Q.unsqueeze(0).expand(nBatch, args.num_var*6, args.num_var*6).to(device)  #could also add some trainable parameters in Q
        
        if self.training and self.use_dcbf:
            e = Variable(torch.Tensor()).to(device)
            x = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)
            out = torch.mean(torch.abs(x + q))  # loss
        else:
            x = cvx_solver(Q[0].detach().double(), q[0].detach().double(), G[0].detach().double(), h[0].detach().double())
            out = []
            for i in range(3*args.num_var):
                out.append(x[i])
            out = np.array(out)
            out = torch.tensor(out).float().to(device)
            out = out.view(args.num_var, 3)

        return out
    
    def dCBF(self, y, w_ref):
        # get the coe of controls
        nBatch = y.shape[0]
        u_rt = []
        for i in range(args.num_var):
            temp_net = copy.deepcopy(self.net)
            if i > 0:
                temp_net[2].weight[:,0:i] = torch.zeros_like(temp_net[2].weight[:,0:i])
            temp_net[2].weight[:,i] = torch.ones_like(temp_net[2].weight[:,i])
            temp_net[2].weight[:,i+1:] = torch.zeros_like(temp_net[2].weight[:,i+1:])
            temp_net[2].bias[:] = torch.zeros_like(temp_net[2].bias[:])
            u_rt.append(temp_net(y))
        
        # get the remaining values
        temp_net =copy.deepcopy(self.net)
        for i in range(args.num_var):
            temp_net[2].weight[:,i] = torch.zeros_like(temp_net[2].weight[:,i])

        bias_rt = temp_net(y)
        
        b = 0.5*(y[0, 0] + y[0, 1]) - y[0,2]
        Lfb = 0.5*(bias_rt[0,0] + bias_rt[0,1]) - bias_rt[0,2]
        Lgbu = []
        for i in range(args.num_var):
            Lgbu1 = torch.reshape(0.5*u_rt[i][0,0], (nBatch,1)) 
            Lgbu2 = torch.reshape(0.5*u_rt[i][0,1], (nBatch,1))
            Lgbu3 = torch.reshape(-u_rt[i][0,2], (nBatch,1))
            Lgbu.append(-Lgbu1)
            Lgbu.append(-Lgbu2)
            Lgbu.append(-Lgbu3)
        G = torch.cat(Lgbu, dim = 1)
        G = torch.reshape(G, (nBatch, 1, args.num_var*3)).to(device)
        h = (torch.reshape((Lfb + 1*b), (nBatch, 1))).to(device)


        # q = -w_ref.unsqueeze(0)
        q = torch.reshape(-w_ref.transpose(1,0).flatten(), (nBatch, args.num_var*3))
        Q = Variable(torch.eye(args.num_var*3))
        Q = Q.unsqueeze(0).expand(nBatch, args.num_var*3, args.num_var*3).to(device)
        x = cvx_solver(Q[0].double(), q[0].double(), G[0].double(), h[0].double())
        
        out = []
        for i in range(3*args.num_var):
            out.append(x[i])
        x = np.array(out)
        x = torch.tensor(x).float().to(device)
        
        
        x = x.view(args.num_var, 3).transpose(1,0)

        return x





if __name__ == '__main__':

    ii = 2

    func = ODEFunc().to(device)

    func.load_state_dict(torch.load("./logs/model0/model_node_itr1560.pth"))

    func.use_dcbf = False
    y0 = fval0
    pred_y_unsafe = y0.unsqueeze(0)
    tt = torch.tensor([0., t[1]]).to(device)
    with torch.no_grad():
        for i in range(args.data_size-1):
            pred = odeint(func, y0, tt)
            pred_y_unsafe = torch.cat([pred_y_unsafe, pred[-1,:,:].unsqueeze(0)], dim = 0)
            y0 = pred[-1,:,:]
            print('Iter {:04d}'.format(i))
    loss = torch.mean(torch.abs(pred_y_unsafe - fval))
    print('Iter {:04d} | Total Loss {:.6f}'.format(0, loss.item()))

    func.use_dcbf = True
    y0 = fval0
    pred_y = y0.unsqueeze(0)
    tt = torch.tensor([0., t[1]]).to(device)
    with torch.no_grad():
        for i in range(args.data_size-1):
            pred = odeint(func, y0, tt)
            pred_y = torch.cat([pred_y, pred[-1,:,:].unsqueeze(0)], dim = 0)
            y0 = pred[-1,:,:]
            print('Iter {:04d}'.format(i))
    loss = torch.mean(torch.abs(pred_y - fval))
    print('Iter {:04d} | Total Loss {:.6f}'.format(0, loss.item()))

    func.use_hidden_layer = True
    func.saved_param = func.net[0].weight[0:args.num_var,:].clone().detach()
    with torch.no_grad():
        tt = torch.tensor([0., t[1]]).to(device)
        y0 = fval0
        b2 = 0.5*(y0[0,0] + y0[0,1]) - y0[0,2]
        pred_y2 = y0.unsqueeze(0)
        sat2 = b2.unsqueeze(0)
        func.eval()
        for i in range(args.data_size-1):  #range(73):#args.data_size-1
            print(i)
            ref_param = func.net[0].weight[0:args.num_var,:].clone().detach()
            control_param = func.dCBF_hidden(y0, ref_param)
            func.saved_param = func.saved_param + control_param*t[1]
            pred = odeint(func, y0, tt)
            pred_y2 = torch.cat([pred_y2, pred[-1,:,:].unsqueeze(0)], dim = 0)
            y0 = pred[-1,:,:]
            b2 = 0.5*(y0[0,0] + y0[0,1]) - y0[0,2]
            sat2 = torch.cat([sat2, b2.unsqueeze(0)], dim = 0)

        loss = torch.mean(torch.abs(pred_y2 - fval))
        print('Total Loss {:.6f}'.format(loss.item()))
        print('min sat {:.6f}'.format(torch.min(sat2)))

    visualize(fval, pred_y, pred_y2, pred_y_unsafe, ii)
    
            

    