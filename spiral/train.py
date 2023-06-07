import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from cvxopt import solvers, matrix
from torch.autograd import Variable

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
parser.add_argument('--consider_noise', action='store_true')  #add noise
parser.add_argument('--hard_truncate', action='store_true')   #hard truncate method
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

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
    obs_x, obs_y = obs[0,0], obs[0,1]  # first obstacle
    b = (y[:,0] - obs_x)**2 + (y[:,1] - obs_y)**2 - 0.2**2
    Lfb = 2*(y[:,0] - obs_x)*(-0.1*y[:,0]**3) + 2*(y[:,1] - obs_y)*(-0.1*y[:,1]**3)
    Lgbu1 = torch.reshape(2*(y[:,0] - obs_x)*y[:,1]**3, (nBatch,1))  
    Lgbu2 = torch.reshape(2*(y[:,1] - obs_y)*y[:,0]**3, (nBatch,1))
    G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
    G = torch.reshape(G, (nBatch, 1, 2)).to(device)
    h = (torch.reshape((Lfb + 30*b), (nBatch, 1))).to(device) 

    obs_x, obs_y = obs[1,0]-0.07, obs[1,1]  # second obstacle
    b = (y[:,0] - obs_x)**2 + (y[:,1] - obs_y)**2 - 0.2**2
    Lfb = 2*(y[:,0] - obs_x)*(-0.1*y[:,0]**3) + 2*(y[:,1] - obs_y)*(-0.1*y[:,1]**3)
    Lgbu1 = torch.reshape(2*(y[:,0] - obs_x)*y[:,1]**3, (nBatch,1)) 
    Lgbu2 = torch.reshape(2*(y[:,1] - obs_y)*y[:,0]**3, (nBatch,1))
    G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
    G1 = torch.reshape(G1, (nBatch, 1, 2)).to(device)
    h1 = (torch.reshape((Lfb + 30*b), (nBatch, 1))).to(device)

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
    fig = plt.figure(figsize=(8, 4), facecolor='white')
    ax_phase = fig.add_subplot(121, frameon=False)
    ax_vecfield = fig.add_subplot(122, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:
        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
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
            # nn.Tanh(),           
            nn.Tanhshrink(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)


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
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        
        if args.hard_truncate:  #hard truncate method
            seq = batch_y.shape[0]
            nbat = batch_y.shape[1]
            pred_y = []
            for j in range(nbat):
                y0 = batch_y0[j,:,:]
                pred_yj = y0.unsqueeze(0)
                for i in range(seq - 1):
                    pred = odeint(func, y0, tt)
                    #check hard constraint
                    obs_x, obs_y = obs[0,0], obs[0,1]
                    b = (pred[-1,:,0] - obs_x)**2 + (pred[-1,:,1] - obs_y)**2 - 0.2**2
                    if b < 0:
                        theta = torch.atan2(pred[-1,:,1] - obs_y, pred[-1,:,0] - obs_x)
                        pred[-1,:,0] = 0.2*torch.cos(theta) + obs_x 
                        pred[-1,:,1] = 0.2*torch.sin(theta) + obs_y
                    obs_x, obs_y = obs[1,0], obs[1,1]
                    b = (pred[-1,:,0] - obs_x)**2 + (pred[-1,:,1] - obs_y)**2 - 0.2**2
                    if b < 0:
                        theta = torch.atan2(pred[-1,:,1] - obs_y, pred[-1,:,0] - obs_x)
                        pred[-1,:,0] = 0.2*torch.cos(theta) + obs_x 
                        pred[-1,:,1] = 0.2*torch.sin(theta) + obs_y
                    #end check
                    pred_yj = torch.cat([pred_yj, pred[-1,:,:].unsqueeze(0)], dim = 0)
                    y0 = pred[-1,:,:]
                pred_y.append(pred_yj.unsqueeze(0))
            pred_y = torch.cat(pred_y, dim=0)
            pred_y = torch.transpose(pred_y, 0, 1)
        else:
            pred_y = odeint(func, batch_y0, batch_t).to(device)

        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                if args.hard_truncate:
                    tt = torch.tensor([0., 0.025]).to(device)
                    y0 = true_y0
                    pred_y = y0.unsqueeze(0)
                    func.eval()
                    for i in range(args.data_size-1):
                        pred = odeint(func, y0, tt)
                        #check hard constraint
                        obs_x, obs_y = obs[0,0], obs[0,1]
                        b = (pred[-1,:,0] - obs_x)**2 + (pred[-1,:,1] - obs_y)**2 - 0.2**2
                        if b < 0:
                            theta = torch.atan2(pred[-1,:,1] - obs_y, pred[-1,:,0] - obs_x)
                            pred[-1,:,0] = 0.2*torch.cos(theta) + obs_x 
                            pred[-1,:,1] = 0.2*torch.sin(theta) + obs_y
                        obs_x, obs_y = obs[1,0], obs[1,1]
                        b = (pred[-1,:,0] - obs_x)**2 + (pred[-1,:,1] - obs_y)**2 - 0.2**2
                        if b < 0:
                            theta = torch.atan2(pred[-1,:,1] - obs_y, pred[-1,:,0] - obs_x)
                            pred[-1,:,0] = 0.2*torch.cos(theta) + obs_x 
                            pred[-1,:,1] = 0.2*torch.sin(theta) + obs_y
                        #end check
                        pred_y = torch.cat([pred_y, pred[-1,:,:].unsqueeze(0)], dim = 0)
                        y0 = pred[-1,:,:]
                else:
                    pred_y = odeint(func, true_y0, t)

                loss = torch.mean(torch.abs(pred_y - true_y))
                test_loss = torch.cat([test_loss, loss.unsqueeze(0).unsqueeze(0)], dim = 0)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                
            torch.save(func.state_dict(), "./logs/model/model_node_itr" + format(ii, '02d') + ".pth")
            ii += 1
        end = time.time()
    data = {'test':test_loss.cpu().numpy()}
    import pickle
    output = open('./logs/test_loss.pkl', 'wb')
    pickle.dump(data, output)
    output.close()

    