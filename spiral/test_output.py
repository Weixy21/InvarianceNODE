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

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--consider_noise', action='store_true')
parser.add_argument('--num_var', type=int, default=1)
parser.add_argument('--dcbf_in_loss', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)
real_A = true_A
obs = torch.tensor([[0, 1.3],[-1.03, 0]]).to(device)

def cvx_solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    return sol['x']

def update_A(true_A, y):
    nBatch = 1
    obs_x, obs_y = obs[0,0], obs[0,1]
    b = (y[:,0] - obs_x)**2 + (y[:,1] - obs_y)**2 - 0.2**2
    Lfb = 2*(y[:,0] - obs_x)*(-0.1*y[:,0]**3) + 2*(y[:,1] - obs_y)*(-0.1*y[:,1]**3)
    Lgbu1 = torch.reshape(2*(y[:,0] - obs_x)*y[:,1]**3, (nBatch,1)) 
    Lgbu2 = torch.reshape(2*(y[:,1] - obs_y)*y[:,0]**3, (nBatch,1))
    G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
    G = torch.reshape(G, (nBatch, 1, 2)).to(device)
    h = (torch.reshape((Lfb + 10*b), (nBatch, 1))).to(device)

    obs_x, obs_y = obs[1,0], obs[1,1]
    b = (y[:,0] - obs_x)**2 + (y[:,1] - obs_y)**2 - 0.2**2
    Lfb = 2*(y[:,0] - obs_x)*(-0.1*y[:,0]**3) + 2*(y[:,1] - obs_y)*(-0.1*y[:,1]**3)
    Lgbu1 = torch.reshape(2*(y[:,0] - obs_x)*y[:,1]**3, (nBatch,1)) 
    Lgbu2 = torch.reshape(2*(y[:,1] - obs_y)*y[:,0]**3, (nBatch,1))
    G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
    G1 = torch.reshape(G1, (nBatch, 1, 2)).to(device)
    h1 = (torch.reshape((Lfb + 10*b), (nBatch, 1))).to(device)

    G = torch.cat([G,G1], dim = 1).to(device)
    h = torch.cat([h,h1], dim = 1).to(device)
    
    q = -torch.tensor([[-2.0, 2.0]]).to(device)
    Q = Variable(torch.eye(2))
    Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(device)
    x = cvx_solver(Q[0].double(), q[0].double(), G[0].double(), h[0].double())
    x = np.array([x[0], x[1]])
    x = torch.tensor(x).float().to(device)
    
    real_A = true_A.clone()
    real_A[1,0] = x[0]
    real_A[0,1] = x[1]
    return real_A

class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y**3, real_A)

with torch.no_grad():
    tt = torch.tensor([0., 0.025]).to(device)
    y0 = true_y0
    true_y = y0.unsqueeze(0)
    for i in range(args.data_size-1):
        real_A = update_A(true_A, y0)
        pred = odeint(Lambda(), y0, tt, method='dopri5')
        true_y = torch.cat([true_y, pred[-1,:,:].unsqueeze(0)], dim = 0)
        y0 = pred[-1,:,:]
        print(i)
    if args.consider_noise:
        true_y += 0.1*(0.5 - torch.rand_like(true_y))


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:

    makedirs('logs/test')
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_phase1 = fig.add_subplot(131, frameon=False)
    ax_phase2 = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y1, pred_y2, cat1, cat2, itr):

    if args.viz:

        ax_phase1.cla()
        ax_phase1.set_title('(a) neural ODE')
        ax_phase1.set_xlabel('$x$')
        ax_phase1.set_ylabel('$y$')
        theta = np.linspace(0., 2*np.pi, 200)
        xx = 0.2*np.cos(theta) + obs.cpu().numpy()[0,0]
        yy = 0.2*np.sin(theta) + obs.cpu().numpy()[0,1]
        ax_phase1.plot(xx, yy, 'r-')
        ax_phase1.fill(xx, yy, 'r-')
        xx = 0.2*np.cos(theta) + obs.cpu().numpy()[1,0]
        yy = 0.2*np.sin(theta) + obs.cpu().numpy()[1,1]
        ax_phase1.plot(xx, yy, 'r-')
        ax_phase1.fill(xx, yy, 'r-')
        ax_phase1.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label = 'ground truth')
        ax_phase1.plot(pred_y1.cpu().numpy()[:, 0, 0], pred_y1.cpu().numpy()[:, 0, 1], 'b--', label = 'prediction')
        ax_phase1.set_xlim(-2, 2)
        ax_phase1.set_ylim(-2, 2.4)
        ax_phase1.legend(loc ='upper left')

        ax_phase2.cla()
        ax_phase2.set_title('(b) neural ODE and invariance')
        ax_phase2.set_xlabel('$x$')
        ax_phase2.set_ylabel('$y$')
        theta = np.linspace(0., 2*np.pi, 200)
        xx = 0.2*np.cos(theta) + obs.cpu().numpy()[0,0]
        yy = 0.2*np.sin(theta) + obs.cpu().numpy()[0,1]
        ax_phase2.plot(xx, yy, 'r-')
        ax_phase2.fill(xx, yy, 'r-')
        xx = 0.2*np.cos(theta) + obs.cpu().numpy()[1,0]
        yy = 0.2*np.sin(theta) + obs.cpu().numpy()[1,1]
        ax_phase2.plot(xx, yy, 'r-')
        ax_phase2.fill(xx, yy, 'r-')
        ax_phase2.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label = 'ground truth')
        ax_phase2.plot(pred_y2.cpu().numpy()[:, 0, 0], pred_y2.cpu().numpy()[:, 0, 1], 'b--', label = 'prediction')
        ax_phase2.set_xlim(-2, 2)
        ax_phase2.set_ylim(-2, 2.4)
        ax_phase2.legend()

        ax_vecfield.cla()
        ax_vecfield.set_title('(c) satisfaction portrait')
        ax_vecfield.set_xlabel('$t/s$')
        ax_vecfield.set_ylabel('$h(x)$')
        ax_vecfield.plot(t.cpu().numpy(), cat1.cpu().numpy(), 'g-', label = 'neural ODE')
        ax_vecfield.plot(t.cpu().numpy(), cat2.cpu().numpy(), 'b-', label = 'neural ODE and invariance')
        ax_vecfield.plot([0, 25.], [0,0], 'k--', label = 'satisfaction boundary')
        ax_vecfield.set_ylim(-0.2, 0.4)
        ax_vecfield.set_xlim(0, 5.)
        ax_vecfield.legend(loc ='upper right')

        fig.tight_layout()
        
        plt.savefig('logs/test/spiral_tr{:03d}.pdf'.format(itr))
        plt.draw()
        plt.pause(0.001)



class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        self.use_dcbf = True
        self.pa = Parameter(torch.ones(1))
        self.pb = Parameter(torch.ones(1))

    def forward(self, t, y):
        if self.use_dcbf:
            w_ref = self.net[2].weight[:,0:args.num_var].clone().detach()
            if self.training:
                out = self.dCBF(y, w_ref)
            else:
                new_w = self.dCBF(y, w_ref)
                temp_net = copy.deepcopy(self.net)
                temp_net[2].weight[:,0:args.num_var] = new_w            
                out = temp_net(y**3)

            return out
        else:
            if args.dcbf_in_loss:
                w_ref = self.net[2].weight[:,0:args.num_var].clone().detach()
                new_w = self.dCBF(y, w_ref)
                temp_net = copy.deepcopy(self.net)
                for param in temp_net.parameters():
                    param.requires_grad = False
                temp_net[2].weight[:,0:args.num_var] = new_w            
                out = temp_net(y**3)
                return out
            else:
                return self.net(y**3)
    
    def dCBF(self, y, w_ref):
        # get the coe of controls
        if self.training and self.use_dcbf:
            y = y.view(args.batch_time*args.batch_size, -1)
        nBatch = y.shape[0]
        u_rt = []
        for i in range(args.num_var):
            temp_net = copy.deepcopy(self.net)
            for param in temp_net.parameters():
                param.requires_grad = False
            if i > 0:
                temp_net[2].weight[:,0:i] = torch.zeros_like(temp_net[2].weight[:,0:i])
            temp_net[2].weight[:,i] = torch.ones_like(temp_net[2].weight[:,i])
            temp_net[2].weight[:,i+1:] = torch.zeros_like(temp_net[2].weight[:,i+1:])
            temp_net[2].bias[:] = torch.zeros_like(temp_net[2].bias[:])
            u_rt.append(temp_net(y**3).detach())
        
        # get the remaining values
        temp_net =copy.deepcopy(self.net)
        for param in temp_net.parameters():
            param.requires_grad = False
        for i in range(args.num_var):
            temp_net[2].weight[:,i] = torch.zeros_like(temp_net[2].weight[:,i])

        bias_rt = temp_net(y**3).detach()
        
        #first obstacle
        obs_x, obs_y = obs[0,0], obs[0,1]
        b = (y[:,0] - obs_x)**2 + (y[:,1] - obs_y)**2 - 0.2**2
        Lfb = 2*(y[:,0] - obs_x)*bias_rt[:,0] + 2*(y[:,1] - obs_y)*bias_rt[:,1]
        Lgbu = []
        for i in range(args.num_var):
            Lgbu1 = torch.reshape(2*(y[:,0] - obs_x)*u_rt[i][:,0], (nBatch,1)) 
            Lgbu2 = torch.reshape(2*(y[:,1] - obs_y)*u_rt[i][:,1], (nBatch,1))
            Lgbu.append(-Lgbu1)
            Lgbu.append(-Lgbu2)
        G = torch.cat(Lgbu, dim = 1)
        G = torch.reshape(G, (nBatch, 1, args.num_var*2)).to(device)
        h = (torch.reshape((Lfb + (40*torch.sigmoid(self.pa)+10)*b), (nBatch, 1))).to(device)

        #second obstacle
        obs_x, obs_y = obs[1,0], obs[1,1]
        b = (y[:,0] - obs_x)**2 + (y[:,1] - obs_y)**2 - 0.2**2
        Lfb = 2*(y[:,0] - obs_x)*bias_rt[:,0] + 2*(y[:,1] - obs_y)*bias_rt[:,1]
        Lgbu = []
        for i in range(args.num_var):
            Lgbu1 = torch.reshape(2*(y[:,0] - obs_x)*u_rt[i][:,0], (nBatch,1)) 
            Lgbu2 = torch.reshape(2*(y[:,1] - obs_y)*u_rt[i][:,1], (nBatch,1))
            Lgbu.append(-Lgbu1)
            Lgbu.append(-Lgbu2)
        G1 = torch.cat(Lgbu, dim = 1)
        G1 = torch.reshape(G1, (nBatch, 1, args.num_var*2)).to(device)
        h1 = (torch.reshape((Lfb + (40*torch.sigmoid(self.pb)+10)*b), (nBatch, 1))).to(device)

        G = torch.cat([G,G1], dim = 1).to(device)
        h = torch.cat([h,h1], dim = 1).to(device)

        q = -w_ref.transpose(1,0).flatten().expand(nBatch, args.num_var*2).to(device)
        Q = Variable(torch.eye(args.num_var*2))
        Q = Q.unsqueeze(0).expand(nBatch, args.num_var*2, args.num_var*2).to(device)  #could also add some trainable parameters in Q
        
        if self.training and self.use_dcbf:
            e = Variable(torch.Tensor()).to(device)
            x = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)
            out = torch.mean(torch.abs(x + q))  # loss
        else:
            x = cvx_solver(Q[0].detach().double(), q[0].detach().double(), G[0].detach().double(), h[0].detach().double())
            out = []
            for i in range(2*args.num_var):
                out.append(x[i])
            out = np.array(out)
            out = torch.tensor(out).float().to(device)
            out = out.view(args.num_var, 2).transpose(1,0)

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

    ii = 100
    
    #without invariance
    func = ODEFunc().to(device)
    func.load_state_dict(torch.load("./logs/model/model_node_itr2000.pth")) 
    func.use_dcbf = False
    with torch.no_grad():
        tt = torch.tensor([0., 0.025]).to(device)
        y0 = true_y0
        obs_x, obs_y = obs[0,0], obs[0,1]
        b1 = (y0[0,0] - obs_x)**2 + (y0[0,1] - obs_y)**2 - 0.2**2
        obs_x, obs_y = obs[1,0], obs[1,1]
        b2 = (y0[0,0] - obs_x)**2 + (y0[0,1] - obs_y)**2 - 0.2**2
        pred_y1 = y0.unsqueeze(0)
        sat1 = torch.min(b1, b2).unsqueeze(0)
        func.eval()
        for i in range(args.data_size-1):
            pred = odeint(func, y0, tt)
            pred_y1 = torch.cat([pred_y1, pred[-1,:,:].unsqueeze(0)], dim = 0)
            y0 = pred[-1,:,:]
            obs_x, obs_y = obs[0,0], obs[0,1]
            b1 = (y0[0,0] - obs_x)**2 + (y0[0,1] - obs_y)**2 - 0.2**2
            obs_x, obs_y = obs[1,0], obs[1,1]
            b2 = (y0[0,0] - obs_x)**2 + (y0[0,1] - obs_y)**2 - 0.2**2
            sat1 = torch.cat([sat1, torch.min(b1, b2).unsqueeze(0)], dim = 0)
        
        loss = torch.mean(torch.abs(pred_y1 - true_y))
        print('Total Loss {:.6f}'.format(loss.item()))

    #with invariance
    func.load_state_dict(torch.load("./logs/model/model_node_itr440.pth")) 
    func.use_dcbf = True
    with torch.no_grad():
        tt = torch.tensor([0., 0.025]).to(device)
        y0 = true_y0
        obs_x, obs_y = obs[0,0], obs[0,1]
        b1 = (y0[0,0] - obs_x)**2 + (y0[0,1] - obs_y)**2 - 0.2**2
        obs_x, obs_y = obs[1,0], obs[1,1]
        b2 = (y0[0,0] - obs_x)**2 + (y0[0,1] - obs_y)**2 - 0.2**2
        pred_y2 = y0.unsqueeze(0)
        sat2 = torch.min(b1, b2).unsqueeze(0)
        func.eval()
        for i in range(args.data_size-1):
            pred = odeint(func, y0, tt)
            pred_y2 = torch.cat([pred_y2, pred[-1,:,:].unsqueeze(0)], dim = 0)
            y0 = pred[-1,:,:]
            obs_x, obs_y = obs[0,0], obs[0,1]
            b1 = (y0[0,0] - obs_x)**2 + (y0[0,1] - obs_y)**2 - 0.2**2
            obs_x, obs_y = obs[1,0], obs[1,1]
            b2 = (y0[0,0] - obs_x)**2 + (y0[0,1] - obs_y)**2 - 0.2**2
            sat2 = torch.cat([sat2, torch.min(b1, b2).unsqueeze(0)], dim = 0)
        
        
        loss = torch.mean(torch.abs(pred_y2 - true_y))
        print('Total Loss {:.6f}'.format(loss.item()))
    
    data = {'pred1':pred_y1.cpu().numpy(), 'pred2':pred_y2.cpu().numpy(), 'sat1':sat1.cpu().numpy(), 'sat2':sat2.cpu().numpy()}
    import pickle
    output = open('logs/test/data_tr.pkl', 'wb')
    pickle.dump(data, output)
    output.close()
    
    # import pickle
    # pkl_file = open('./data_tr.pkl','rb')
    # data = pickle.load(pkl_file)
    # pred_y1 = torch.tensor(data['pred1']).float() # batch, seq, data
    # pred_y2 = torch.tensor(data['pred2']).float()
    # sat1 = torch.tensor(data['sat1']).float()
    # sat2 = torch.tensor(data['sat2']).float()
    # pkl_file.close()
    
    # import pdb; pdb.set_trace()
    visualize(true_y, pred_y1, pred_y2, sat1, sat2, ii)
            
    end = time.time()

    