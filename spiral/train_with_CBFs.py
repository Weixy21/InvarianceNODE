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
parser.add_argument('--niters', type=int, default=2100)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--consider_noise', action='store_true')
parser.add_argument('--num_var', type=int, default=1)  # number of parameters the invariance is propagated to, 2x
parser.add_argument('--dcbf_in_loss', action='store_true') # consider the CBF effect in the loss
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

#CBFs for safe spiral curve ground truth
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

# generate safe spiral curve ground truth using CBFs
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
    makedirs('./logs/png')
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g*')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('(b) neural ODE and invariance') 
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        theta = np.linspace(0., 2*np.pi, 200)
        xx = 0.2*np.cos(theta) + obs.cpu().numpy()[0,0]
        yy = 0.2*np.sin(theta) + obs.cpu().numpy()[0,1]
        ax_phase.plot(xx, yy, 'r-')
        ax_phase.fill(xx, yy, 'r-')

        xx = 0.2*np.cos(theta) + obs.cpu().numpy()[1,0]
        yy = 0.2*np.sin(theta) + obs.cpu().numpy()[1,1]
        ax_phase.plot(xx, yy, 'r-')
        ax_phase.fill(xx, yy, 'r-')

        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label = 'ground truth')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--', label = 'prediction')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)
        ax_phase.legend()

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('./logs/png/{:03d}'.format(itr))
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

    ii = 0
    makedirs('./logs/model')
    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    tt = torch.tensor([0., 0.025]).to(device)

    test_loss = torch.tensor([[0.]]).to(device)

    for itr in range(1, args.niters + 1):
        # select training mode, first train neural ODE
        func.use_dcbf = False
        func.train()
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        if args.dcbf_in_loss:
            seq = batch_y.shape[0]
            nbat = batch_y.shape[1]
            pred_y = []
            for j in range(nbat):
                y0 = batch_y0[j,:,:]
                pred_yj = y0.unsqueeze(0)
                for i in range(seq - 1):
                    pred = odeint(func, y0, tt)
                    pred_yj = torch.cat([pred_yj, pred[-1,:,:].unsqueeze(0)], dim = 0)
                    y0 = pred[-1,:,:]
                pred_y.append(pred_yj.unsqueeze(0))
            pred_y = torch.cat(pred_y, dim=0)
            pred_y = torch.transpose(pred_y, 0, 1)
        else:
            pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward(retain_graph=True)

        #next train CBFs
        # freeze network parameters:
        for param in func.parameters():
            param.requires_grad = False
        func.pa.requires_grad = True
        func.pb.requires_grad = True

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        # change training mode to dCBF
        func.use_dcbf = True    
        loss2 = func(batch_t, pred_y).to(device)
        loss2.backward()

        optimizer.step()

        # unfreeze network parameters:
        for param in func.parameters():
            param.requires_grad = True

        if itr % args.test_freq == 0:
            with torch.no_grad():

                tt = torch.tensor([0., 0.025]).to(device)
                y0 = true_y0
                pred_y = y0.unsqueeze(0)
                func.eval()
                for i in range(args.data_size-1):
                    pred = odeint(func, y0, tt)
                    pred_y = torch.cat([pred_y, pred[-1,:,:].unsqueeze(0)], dim = 0)
                    y0 = pred[-1,:,:]
                
                loss = torch.mean(torch.abs(pred_y - true_y))
                test_loss = torch.cat([test_loss, loss.unsqueeze(0).unsqueeze(0)], dim = 0)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                func.use_dcbf = False
                visualize(true_y, pred_y, func, ii)
                
            torch.save(func.state_dict(), "./logs/model/model_node_itr" + format(ii, '02d') + ".pth")
            ii += 1
        end = time.time()
    data = {'test':test_loss.cpu().numpy()}
    import pickle
    output = open('./logs/model/test_loss.pkl', 'wb')
    pickle.dump(data, output)
    output.close()

    