import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from cvxopt import solvers, matrix
import copy

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
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

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    if args.consider_noise:
        true_y += 0.1*(0.5 - torch.rand_like(true_y))


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('./logs/test')
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    fig = plt.figure(figsize=(16, 4), facecolor='white')
    ax_phase1 = fig.add_subplot(141, frameon=False)
    ax_phase2 = fig.add_subplot(142, frameon=False)
    ax_phase3 = fig.add_subplot(143, frameon=False)
    ax_phase4 = fig.add_subplot(144, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y1, pred_y2, pred_y3, pred_y4, obs1, obs2, itr):

    if args.viz:
 
        ax_phase1.cla()
        ax_phase1.set_title('$(a)$ rough CBF, $n_p = 2$')
        theta = np.linspace(0., 2*np.pi, 200)
        xx = 0.2*np.cos(theta) + 0
        yy = 0.2*np.sin(theta) + 1.2
        ax_phase1.plot(xx, yy, 'r-')
        ax_phase1.fill(xx, yy, 'r-')
        xx = 0.2*np.cos(theta) - 0.9
        yy = 0.2*np.sin(theta) + 0
        ax_phase1.plot(xx, yy, 'r-')
        ax_phase1.fill(xx, yy, 'r-')
        ax_phase1.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label = 'ground truth')
        ax_phase1.plot(pred_y1.cpu().numpy()[:, 0, 0], pred_y1.cpu().numpy()[:, 0, 1], 'b--', label = 'prediction')
        ax_phase1.set_xlim(-2, 2)
        ax_phase1.set_ylim(-2, 2.4)
        ax_phase1.legend()
        ax_phase1.set_xlabel('$x$')
        ax_phase1.set_ylabel('$y$')


        ax_phase2.cla()
        ax_phase2.set_title('$(b)$ fine-tuned CBF, $n_p = 2$')
        theta = np.linspace(0., 2*np.pi, 200)
        xx = 0.2*np.cos(theta) + 0
        yy = 0.2*np.sin(theta) + 1.2
        ax_phase2.plot(xx, yy, 'r-')
        ax_phase2.fill(xx, yy, 'r-')
        xx = 0.2*np.cos(theta) - 0.9
        yy = 0.2*np.sin(theta) + 0
        ax_phase2.plot(xx, yy, 'r-')
        ax_phase2.fill(xx, yy, 'r-')
        ax_phase2.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label = 'ground truth')
        ax_phase2.plot(pred_y2.cpu().numpy()[:, 0, 0], pred_y2.cpu().numpy()[:, 0, 1], 'b--', label = 'prediction')
        ax_phase2.set_xlim(-2, 2)
        ax_phase2.set_ylim(-2, 2.4)
        ax_phase2.legend()
        ax_phase2.set_xlabel('$x$')
        ax_phase2.set_ylabel('$y$')


        ax_phase3.cla()
        ax_phase3.set_title('$(c)$ fine-tuned CBF, $n_p = 10$')
        theta = np.linspace(0., 2*np.pi, 200)
        xx = 0.2*np.cos(theta) + 0
        yy = 0.2*np.sin(theta) + 1.2
        ax_phase3.plot(xx, yy, 'r-')
        ax_phase3.fill(xx, yy, 'r-')
        xx = 0.2*np.cos(theta) - 0.9
        yy = 0.2*np.sin(theta) + 0
        ax_phase3.plot(xx, yy, 'r-')
        ax_phase3.fill(xx, yy, 'r-')
        ax_phase3.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label = 'ground truth')
        ax_phase3.plot(pred_y3.cpu().numpy()[:, 0, 0], pred_y3.cpu().numpy()[:, 0, 1], 'b--', label = 'prediction')
        ax_phase3.set_xlim(-2, 2)
        ax_phase3.set_ylim(-2, 2.4)
        ax_phase3.legend()
        ax_phase3.set_xlabel('$x$')
        ax_phase3.set_ylabel('$y$')

        ax_phase4.cla()
        ax_phase4.set_title('$(d)$ other spec.')
        theta = np.linspace(0., 2*np.pi, 200)
        xx = 0.2*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + obs1[0]
        yy = 0.2*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + obs1[1]
        ax_phase4.plot(xx, yy, 'r-')
        ax_phase4.fill(xx, yy, 'r-')
        xx = 0.2*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + obs2[0]
        yy = 0.2*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + obs2[1]
        ax_phase4.plot(xx, yy, 'r-')
        ax_phase4.fill(xx, yy, 'r-')
        ax_phase4.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label = 'ground truth')
        ax_phase4.plot(pred_y4.cpu().numpy()[:, 0, 0], pred_y4.cpu().numpy()[:, 0, 1], 'b--', label = 'prediction')
        ax_phase4.set_xlim(-2, 2)
        ax_phase4.set_ylim(-2, 2.4)
        ax_phase4.legend()
        ax_phase4.set_xlabel('$x$')
        ax_phase4.set_ylabel('$y$')

        fig.tight_layout()
        plt.savefig('logs/test/spiral{:03d}.pdf'.format(itr))
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
            nn.Linear(2, 50),
            #nn.Tanh(),
            nn.Tanhshrink(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        self.use_dcbf = True
        self.p = 10
        self.obs1 = []
        self.obs2 = []
        self.circ = True
    def forward(self, t, y):
        
        if self.use_dcbf:
            w_ref = self.net[2].weight[:,0:args.num_var]
            new_w = self.dCBF(y, w_ref)

            temp_net = copy.deepcopy(self.net)
            temp_net[2].weight[:,0:args.num_var] = new_w #.squeeze(0)
            
            out = temp_net(y**3)
        else:
            out = self.net(y**3)
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
            u_rt.append(temp_net(y**3))
        
        # get the remaining values
        temp_net =copy.deepcopy(self.net)
        for i in range(args.num_var):
            temp_net[2].weight[:,i] = torch.zeros_like(temp_net[2].weight[:,i])

        bias_rt = temp_net(y**3)
        
        if self.circ:
            obs_x, obs_y = self.obs1[0], self.obs1[1]
            b = (y[0,0] - obs_x)**2 + (y[0,1] - obs_y)**2 - 0.2**2
            Lfb = 2*(y[0,0] - obs_x)*bias_rt[0,0] + 2*(y[0,1] - obs_y)*bias_rt[0,1]
            Lgbu = []
            for i in range(args.num_var):
                Lgbu1 = torch.reshape(2*(y[0,0] - obs_x)*u_rt[i][0,0], (nBatch,1)) 
                Lgbu2 = torch.reshape(2*(y[0,1] - obs_y)*u_rt[i][0,1], (nBatch,1))
                Lgbu.append(-Lgbu1)
                Lgbu.append(-Lgbu2)
            G = torch.cat(Lgbu, dim = 1)
            G = torch.reshape(G, (nBatch, 1, args.num_var*2)).to(device)
            h = (torch.reshape((Lfb + self.p*b), (nBatch, 1))).to(device)

            obs_x, obs_y = self.obs2[0], self.obs2[1]
            b = (y[0,0] - obs_x)**2 + (y[0,1] - obs_y)**2 - 0.2**2
            Lfb = 2*(y[0,0] - obs_x)*bias_rt[0,0] + 2*(y[0,1] - obs_y)*bias_rt[0,1]
            Lgbu = []
            for i in range(args.num_var):
                Lgbu1 = torch.reshape(2*(y[0,0] - obs_x)*u_rt[i][0,0], (nBatch,1)) 
                Lgbu2 = torch.reshape(2*(y[0,1] - obs_y)*u_rt[i][0,1], (nBatch,1))
                Lgbu.append(-Lgbu1)
                Lgbu.append(-Lgbu2)
            G1 = torch.cat(Lgbu, dim = 1)
            G1 = torch.reshape(G1, (nBatch, 1, args.num_var*2)).to(device)
            h1 = (torch.reshape((Lfb + self.p*b), (nBatch, 1))).to(device)
        else:
            obs_x, obs_y = self.obs1[0], self.obs1[1]
            b = (y[0,0] - obs_x)**4 + (y[0,1] - obs_y)**4 - 0.2**4
            Lfb = 4*(y[0,0] - obs_x)**3*bias_rt[0,0] + 4*(y[0,1] - obs_y)**3*bias_rt[0,1]
            Lgbu = []
            for i in range(args.num_var):
                Lgbu1 = torch.reshape(4*(y[0,0] - obs_x)**3*u_rt[i][0,0], (nBatch,1)) 
                Lgbu2 = torch.reshape(4*(y[0,1] - obs_y)**3*u_rt[i][0,1], (nBatch,1))
                Lgbu.append(-Lgbu1)
                Lgbu.append(-Lgbu2)
            G = torch.cat(Lgbu, dim = 1)
            G = torch.reshape(G, (nBatch, 1, args.num_var*2)).to(device)
            h = (torch.reshape((Lfb + self.p*b), (nBatch, 1))).to(device)

            obs_x, obs_y = self.obs2[0], self.obs2[1]
            b = (y[0,0] - obs_x)**4 + (y[0,1] - obs_y)**4 - 0.2**4
            Lfb = 4*(y[0,0] - obs_x)**3*bias_rt[0,0] + 4*(y[0,1] - obs_y)**3*bias_rt[0,1]
            Lgbu = []
            for i in range(args.num_var):
                Lgbu1 = torch.reshape(4*(y[0,0] - obs_x)**3*u_rt[i][0,0], (nBatch,1)) 
                Lgbu2 = torch.reshape(4*(y[0,1] - obs_y)**3*u_rt[i][0,1], (nBatch,1))
                Lgbu.append(-Lgbu1)
                Lgbu.append(-Lgbu2)
            G1 = torch.cat(Lgbu, dim = 1)
            G1 = torch.reshape(G1, (nBatch, 1, args.num_var*2)).to(device)
            h1 = (torch.reshape((Lfb + self.p*b), (nBatch, 1))).to(device)

        G = torch.cat([G,G1], dim = 1).to(device)
        h = torch.cat([h,h1], dim = 1).to(device)

        q = torch.reshape(-w_ref.transpose(1,0).flatten(), (nBatch, args.num_var*2))
        Q = Variable(torch.eye(args.num_var*2))
        Q = Q.unsqueeze(0).expand(nBatch, args.num_var*2, args.num_var*2).to(device)
        x = cvx_solver(Q[0].double(), q[0].double(), G[0].double(), h[0].double())
        
        out = []
        for i in range(2*args.num_var):
            out.append(x[i])
        x = np.array(out)
        x = torch.tensor(x).float().to(device)
        
        
        x = x.view(args.num_var, 2).transpose(1,0)

        return x





if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    
    func.load_state_dict(torch.load("./logs/model/model_node_itr1980.pth")) 
    
    # conservative CBFs, 1x2 parameters
    func.obs1 = torch.tensor([0, 1.2]).to(device)
    func.obs2 = torch.tensor([-0.9, 0]).to(device)
    func.use_dcbf = True
    y0 = true_y0
    pred_y1 = y0.unsqueeze(0)
    tt = torch.tensor([0., 0.025]).to(device)
    with torch.no_grad():
        for i in range(args.data_size-1):
            pred = odeint(func, y0, tt)
            pred_y1 = torch.cat([pred_y1, pred[-1,:,:].unsqueeze(0)], dim = 0)
            y0 = pred[-1,:,:]
            print('Iter {:04d}'.format(i))
    loss = torch.mean(torch.abs(pred_y1 - true_y))
    print('Iter {:04d} | Total Loss {:.6f}'.format(0, loss.item()))
    
    #non-conservative CBFs, 1x2 parameters
    func.p = 20
    y0 = true_y0
    pred_y2 = y0.unsqueeze(0)
    tt = torch.tensor([0., 0.025]).to(device)
    with torch.no_grad():
        for i in range(args.data_size-1):
            pred = odeint(func, y0, tt)
            pred_y2 = torch.cat([pred_y2, pred[-1,:,:].unsqueeze(0)], dim = 0)
            y0 = pred[-1,:,:]
            print('Iter {:04d}'.format(i))
    loss = torch.mean(torch.abs(pred_y2 - true_y))
    print('Iter {:04d} | Total Loss {:.6f}'.format(0, loss.item()))
    
    #non-conservative CBFs, 5x2 parameters
    args.num_var = 5
    y0 = true_y0
    pred_y3 = y0.unsqueeze(0)
    tt = torch.tensor([0., 0.025]).to(device)
    with torch.no_grad():
        for i in range(args.data_size-1):
            pred = odeint(func, y0, tt)
            pred_y3 = torch.cat([pred_y3, pred[-1,:,:].unsqueeze(0)], dim = 0)
            y0 = pred[-1,:,:]
            print('Iter {:04d}'.format(i))
    loss = torch.mean(torch.abs(pred_y3 - true_y))
    print('Iter {:04d} | Total Loss {:.6f}'.format(0, loss.item()))

    #non-conservative CBFs, 1x2 parameters, superellipse obstacles
    func.obs1 = torch.tensor([-0.1, -1.1]).to(device)
    func.obs2 = torch.tensor([0.8, 0.1]).to(device)
    args.num_var = 1
    func.circ = False
    y0 = true_y0
    pred_y4 = y0.unsqueeze(0)
    tt = torch.tensor([0., 0.025]).to(device)
    with torch.no_grad():
        for i in range(args.data_size-1):
            pred = odeint(func, y0, tt)
            pred_y4 = torch.cat([pred_y4, pred[-1,:,:].unsqueeze(0)], dim = 0)
            y0 = pred[-1,:,:]
            print('Iter {:04d}'.format(i))
    loss = torch.mean(torch.abs(pred_y4 - true_y))
    print('Iter {:04d} | Total Loss {:.6f}'.format(0, loss.item()))

    data = {'pred1':pred_y1.cpu().numpy(), 'pred2':pred_y2.cpu().numpy(), 'pred3':pred_y3.cpu().numpy(), 'pred4':pred_y4.cpu().numpy()}
    import pickle
    output = open('./logs/test/data.pkl', 'wb')
    pickle.dump(data, output)
    output.close()

    visualize(true_y, pred_y1, pred_y2, pred_y3, pred_y4, func.obs1.cpu().numpy(), func.obs2.cpu().numpy(), ii)
    
            

    