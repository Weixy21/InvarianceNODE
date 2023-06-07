#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint as odeint_sci
import matplotlib.animation as animation
import os
import argparse
import numpy as np
from torch.nn.functional import normalize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from cvxopt import solvers, matrix
import copy
import pickle
plt.style.use('bmh')


from env import map, sensing, search_vehicles


parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--num_var', type=int, default=1)
parser.add_argument('--use_cnn', action='store_true')
parser.add_argument('--use_fc', action='store_true')
parser.add_argument('--use_dcbf', action='store_true')
parser.add_argument('--dcbf_inside', action='store_true')  #this is for use_cnn, inside means invariance after CNN, has the lidar correction plot problem
parser.add_argument('--with_noise', action='store_true')
parser.add_argument('--use_filter', action='store_true')

args = parser.parse_args()


if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = 'cpu'

pkl_file = open('./data/data4.pkl','rb')
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
lidar_test = lidar_bat[-1,:,:,:]
control_test = control_bat[-1,:,:,:]
ego_test = ego_bat[-1,:,:,:]
other_test = other_bat[-1,:,:,:]

control_test0 = control_bat[-1,0,:,:]
lidar_test0 = lidar_bat[-1,0,:,:]
ego_test0 = ego_bat[-1,0,:,:]
other_test0 = other_bat[-1,0,:,:]

#dynamics
def dynamics(y,t):
    dxdt = y[3]*np.cos(y[2])
    dydt = y[3]*np.sin(y[2])
    dttdt =y[4]
    dvdt = y[5]
    return [dxdt, dydt, dttdt, dvdt, 0, 0]


def cvx_solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)

    return sol['x']

##########################################################################env



veh_num = 50
road_num = 4
veh_state = np.zeros((veh_num, 8)) # x, y, theta, v, psi, lane_id, pen_id, desired_spd, the first vehicle is ego
veh_state[0, 0:4] = ego_test0.cpu().numpy()[0,:]#[0, 6, 0, 16]  #2

veh_state[0, 5:7] = [3, 0]   # 2
active_num = 1
pen_id = 1
marker_pos = veh_state[0, 0] + 26

ax, fig, veh, veh_un, road, speed_handle, pred_handle, ego_correct = map(veh_num, road_num)



##################################################model
class ODEFunc(nn.Module):

    def __init__(self, fc_param, conv_filters,
                 dropout= 0.2):
        super(ODEFunc, self).__init__()

        self.net = self.build_mlp(fc_param)
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        self.use_dcbf = False
        self.lidar = lidar_test0
        self.ego = ego_test0
        self.other = other_test0

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
                x = self._fc(self.lidar)
            g = self.net(y)
            u1, u2 = torch.chunk(g[:,:,2:], 2, dim=-1)
            
            if args.use_fc == True:
                sensor = torch.cat([x, self.ego[:,:,2:4]], dim = 2)  # add theta and speed
            else:
                sensor = torch.cat([self.lidar, self.ego[:,:,2:4]], dim = 2)
            out = torch.cat([(u1*sensor).sum(axis = 2).unsqueeze(2), (u2*sensor).sum(axis = 2).unsqueeze(2)], dim = 2) + g[:,:,0:2]
        else:
            if args.use_fc == True:
                x = self._fc(self.lidar)
            g = self.net(y)
            u1, u2 = torch.chunk(g[:,2:], 2, dim=-1)
              
            if args.use_fc == True:
                sensor = torch.cat([x, self.ego[:,2:4]], dim = 1)  # add theta and speed
            else:
                if args.use_dcbf and args.dcbf_inside:   #inside implementation
                    sensor_safe = self.dCBF(y, self.lidar[0,:], g[:,0:2], u1[0,:], u2[0,:], self.ego, self.other)
                else:
                    sensor_safe = self.lidar
                sensor = torch.cat([sensor_safe, self.ego[:,2:4]], dim = 1)
            out = torch.cat([(u1*sensor).sum(axis = 1).unsqueeze(1), (u2*sensor).sum(axis = 1).unsqueeze(1)], dim = 1) + g[:,0:2]
        return out
    
    def dCBF(self, u, I, f, g1_complete, g2_complete, ego, other):
        x, y, theta, v = ego[0,0], ego[0,1], ego[0,2], ego[0,3]*180
        x0, y0, theta0, v0 = other[0,0], other[0,1], other[0,2], other[0,3]
        u1, u2 = u[0,0], u[0,1]
        f1, f2 = f[0,0], f[0,1]
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        r = 15.5
        g1, g2 = g1_complete[0:-2], g2_complete[0:-2]
        state = torch.clone(ego) #to recall the assignment error
        
        f1_state, f2_state = (g1_complete[-2:]*state[0,2:4]).sum(axis = 0), (g2_complete[-2:]*state[0,2:4]).sum(axis = 0)
        dim = g1.shape[0]
    
        b = (x - x0)**2 + (y - (y0 - 12))**2 - r**2
   
        Lfb = 2*(x - x0)*(v*cos_theta - v0) + 2*(y - (y0 - 12))*v*sin_theta
        Lf2b = 2*(v*cos_theta - v0)**2 + 2*(v*sin_theta)**2 \
            + 2*(x - x0)*(cos_theta*u2 - v*sin_theta*u1) \
                + 2*(y - (y0 - 12))*(sin_theta*u2 + v*cos_theta*u1)
        Lf3b = 6*(v*cos_theta - v0)*(u2*cos_theta - v*sin_theta*u1) \
            + 2*(x - x0)*(-sin_theta*u1*u2 + cos_theta*(f2 + f2_state) - u2*sin_theta*u1 - v*cos_theta*u1**2 - v*sin_theta*(f1+f1_state)) \
                + 2*(y - (y0 - 12))*(cos_theta*u1*u2 + sin_theta*(f2 + f2_state) + u2*cos_theta*u1 - v*sin_theta*u1**2 + v*cos_theta*(f1+f1_state))\
                    + 6*v*sin_theta*(u2*sin_theta + v*cos_theta*u1)
        LgLfbnu = (2*(x - x0)*(-v*sin_theta) + 2*(y - (y0 - 12))*v*cos_theta)*g1 \
            + (2*(x - x0)*cos_theta + 2*(y - (y0 - 12))*sin_theta)*g2
        LgLfbnu = LgLfbnu.unsqueeze(0)
        
        k = 2
        b_safe = Lf3b + 3*k*Lf2b + 3*k**2*Lfb + k**3*b
        A_safe = -LgLfbnu

        G = A_safe.to(device)
        h = b_safe.unsqueeze(0).to(device)

        Q = torch.eye(dim).to(device)
        q = -I.unsqueeze(1)

        x = cvx_solver(Q.double(), q.double(), G.double(), h.double())
        out = []
        for i in range(dim):
            out.append(x[i])
        out = np.array(out)
        out = torch.tensor(out).float().to(device)
        out = out.unsqueeze(0)
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
    
    def build_mlp(self, filters, dropout=0.2, no_act_last_layer=True, activation='gelu'):
        if activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'silu':
            activation = nn.SiLU()
        elif activation == 'tanh':
            activation = nn.Tanh()
        else:
            raise NotImplementedError(f'Not supported activation function {activation}')
        modules = nn.ModuleList()
        for i in range(len(filters)-1):
            modules.append(nn.Linear(filters[i], filters[i+1]))
            if not (no_act_last_layer and i == len(filters)-2):
                modules.append(activation)
                # if dropout > 0.:
                #     modules.append(nn.Dropout(p=dropout))

        modules = nn.Sequential(*modules)
        return modules

def CBF_filter(u1, u2, ego, other):
    x, y, theta, v = ego[0,0], ego[0,1], ego[0,2], ego[0,3]*180
    x0, y0, theta0, v0 = other[0,0], other[0,1], other[0,2], other[0,3]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    r = 15.5
    
    b = (x - x0)**2 + (y - (y0 - 12))**2 - r**2

    Lfb = 2*(x - x0)*(v*cos_theta - v0) + 2*(y - (y0 - 12))*v*sin_theta
    Lf2b = 2*(v*cos_theta - v0)**2 + 2*(v*sin_theta)**2
    LgLfbu1 = 2*(x - x0)*(- v*sin_theta) + 2*(y - (y0 - 12))*(v*cos_theta)
    LgLfbu2 = 2*(x - x0)*(cos_theta) + 2*(y - (y0 - 12))*(sin_theta)
    
    
    k = 1 
    b_safe = Lf2b + 2*k*Lfb + k**2*b
    A_safe = np.array([[-LgLfbu1, -LgLfbu2]])

    G = A_safe
    h = np.array([b_safe])

    Q = torch.eye(2).to(device)
    q = torch.tensor(-np.array([[u1], [u2]])).to(device)
    G = torch.tensor(G).to(device)
    h = torch.tensor(h).to(device)
    x = cvx_solver(Q.double(), q.double(), G.double(), h.double())
    out = []
    for i in range(2):
        out.append(x[i])
    out = np.array(out)
    u1, u2 = out[0], out[1]
    return u1, u2

if args.use_cnn or args.use_fc:
    fc_param = [2, 64, 128, 30] 
else:
    fc_param = [2, 64, 256, 512, 206]

#channel x output channels x kernel size x stride x padding
conv_param = [[1, 4, 5, 2, 1], [4, 8, 3, 2, 1], [8, 12, 3, 2, 0]]
func = ODEFunc(fc_param, conv_param).to(device)
func.load_state_dict(torch.load("./logs/model/model_node_itr10.pth"))
tt = torch.tensor([0., 0.1]).to(device)
u0 = torch.clone(control_test0)
if args.use_cnn:
    u0[0,0] = 7.7*u0[0,0]   # initial condition sensitive
else:
    u0[0,0] = 5*u0[0,0]
pred_u = u0.unsqueeze(0)
func.eval()
#######################################################

traj = []
state, safety = [],[]
once = 1
def update(n):
    global pred_u, u0, state, ax, veh, veh_un, road, speed_handle, pred_handle, ego_correct, veh_num, road_num, veh_state, active_num, pen_id, marker_pos, once
    #print(n)
    dT, horizon = 0.1, 15
    if once == 1:
        once = 0
        lane_id = 3

        speed = 13.5
        veh_state[active_num, 0:4] = other_test0.cpu().numpy()[0,:]
        veh_state[active_num, 5:8] = [lane_id, pen_id, speed]
        active_num += 1
        pen_id += 1
        if pen_id > 49:
            pen_id = 1
    
    # update lane id for ego
    lane_pos = [-2, 2, 6, 10]
    dis = np.array((lane_pos - veh_state[0,1])**2)
    min_dis = np.min(dis)
    min_idx = np.where(min_dis == dis)
    veh_state[0,5] = min_idx[0]+1

    #update marker
    if(marker_pos < veh_state[0,0] - 26):
        marker_pos = veh_state[0,0] + 26
    
    indexes,_ = search_vehicles(veh_state, active_num)  #Lidar
    results, dist = sensing(veh_state, indexes, args.with_noise)
    
    ##############################################neural control
    with torch.no_grad():
        if args.use_cnn:
            sensor = torch.tensor(dist/20.).float().unsqueeze(0).to(device)
        else:
            sensor = torch.tensor(dist/200.).float().unsqueeze(0).to(device)

        ego_state = torch.tensor(veh_state[0,0:4]).float().unsqueeze(0).to(device) 
        func.ego = torch.clone(ego_state) #not needed
        func.ego[0,3] = func.ego[0,3]/180
        func.other = torch.tensor(veh_state[1,0:4]).float().unsqueeze(0).to(device)

        if args.use_cnn == True:
            x = sensor.unsqueeze(0)
            sensor_cnn = func._cnn(x)
              
            if args.use_dcbf and not args.dcbf_inside:
                temp_lidar = sensor_cnn.max(dim=-1)[0]
                g = func.net(u0)
                u1, u2 = torch.chunk(g[:,2:], 2, dim=-1)
                func.lidar = func.dCBF(u0, temp_lidar[0,:], g[:,0:2], u1[0,:], u2[0,:], func.ego, func.other)
            else:
                func.lidar = sensor_cnn.max(dim=-1)[0]
        else:
            if args.use_dcbf and not args.dcbf_inside:
                g = func.net(u0)
                u1, u2 = torch.chunk(g[:,2:], 2, dim=-1)
                func.lidar = func.dCBF(u0, sensor[0,:], g[:,0:2], u1[0,:], u2[0,:], func.ego, func.other)
            else:
                func.lidar = sensor
        
        
        pred = odeint(func, u0, tt)
        pred_u = torch.cat([pred_u, pred[-1,:,:].unsqueeze(0)], dim = 0)
        u0 = pred[-1,:,:]
        u1 = pred.cpu().numpy()[-1,0,0]
        u2 = pred.cpu().numpy()[-1,0,1]

        if args.use_filter:
            u1, u2 = CBF_filter(u1, u2, func.ego.cpu().numpy(), func.other.cpu().numpy())

        barrier = (veh_state[0,0] - veh_state[1,0])**2 + (veh_state[0,1] - (veh_state[1,1] - 12))**2 - 15.5**2
        safety.append(barrier)
    ############################################################

    
    traj.append(np.array([veh_state[0,0], veh_state[0,1], veh_state[0,2], veh_state[0,3]]))
    ego_state = veh_state[0,0:4].tolist()
    ego_state.append(u1)
    ego_state.append(u2)
    
    dt = [0,0.1]
    rt = np.float32(odeint_sci(dynamics,ego_state,dt))
    # import pdb; pdb.set_trace()
    veh_state[0,0:4] = rt[1][0:4]

    for i in range(1, active_num, 1):
        veh_state[i, 0] += veh_state[i, 3]*dT
    
    
    cir_x = results[:,0]
    cir_y = results[:,1]
    corner = np.array([[-2.5, -1.0], [-2.5, +1.0], [+2.5, +1.0], [+2.5, -1.0], [-2.5, -1.0]])
    
    if args.use_dcbf:
        cir_x_cor = np.zeros_like(cir_x)
        cir_y_cor = np.zeros_like(cir_y)
        lidar_cor = func.lidar.cpu().numpy()[0,:]
        for jj in np.arange(100):  #recover to coordinates
            cir_x_cor[jj] = 200*lidar_cor[jj]*np.cos(2*np.pi/100*jj)
            cir_y_cor[jj] = 200*lidar_cor[jj]*np.sin(2*np.pi/100*jj)

    for i in range(active_num):
        if i == 0:
            rot = np.array([[np.cos(veh_state[i, 2]), -np.sin(veh_state[i, 2])],[np.sin(veh_state[i, 2]), np.cos(veh_state[i, 2])]])
            rot_corner = rot.dot(corner.transpose()).transpose()
            veh[int(veh_state[i, 6])].set_data(veh_state[i, 0] + rot_corner[:,0], veh_state[i, 1] + rot_corner[:,1])        
            veh_un[int(veh_state[i, 6])].set_data(veh_state[i, 0]  + cir_x, veh_state[i, 1] + cir_y) 
            if args.use_dcbf:
                ego_correct.set_data(veh_state[i, 0]  + cir_x_cor, veh_state[i, 1] + cir_y_cor) 
            speed_handle[int(veh_state[i, 6])].set_position((veh_state[i, 0]-2, veh_state[i, 1]))
            speed_handle[int(veh_state[i, 6])].set_text(f"{veh_state[i,3]:>.2f} m/s")
            
        else:
            veh[int(veh_state[i, 6])].set_data(veh_state[i, 0] + corner[:,0], veh_state[i, 1] + corner[:,1])
            speed_handle[int(veh_state[i, 6])].set_position((veh_state[i, 0]-2, veh_state[i, 1]))
            speed_handle[int(veh_state[i, 6])].set_text(f"{veh_state[i,3]:>.2f} m/s")

    for i in range(road_num + 1):
        road[i].set_data(veh_state[0, 0] + [-40, 40], [4*i - 4, 4*i - 4])
    road[-1].set_data(marker_pos, 13)
    road[-2].set_data(marker_pos, -5)

    i = 0
    while(i < active_num):    
        if veh_state[i, 0] < veh_state[0, 0] - 40:
            for j in range(active_num - i):
                veh_state[i+j, :] = veh_state[i+j+1, :]
            active_num -= 1
            i -= 1
        i += 1
    ax.axis([-24 + veh_state[0, 0], 24 + veh_state[0, 0], -16, 16])
    

# ################batch test
# for iter in range(10000):
#     print(iter)
#     veh_state = np.zeros((veh_num, 8)) # x, y, theta, v, psi, lane_id, pen_id, desired_spd, the first vehicle is ego
#     veh_state[0, 0:4] = ego_test0.cpu().numpy()[0,:]#[0, 6, 0, 16]  #2
#     veh_state[0, 5:7] = [3, 0]   # 2
#     active_num = 1
#     pen_id = 1
#     marker_pos = veh_state[0, 0] + 26

#     u0 = torch.clone(control_test0)
#     if args.use_cnn:
#         u0[0,0] = 7.7*u0[0,0]   # initial condition sensitive
#     else:
#         u0[0,0] = 5*u0[0,0]
#     pred_u = u0.unsqueeze(0)

#     traj = []
#     state, safety = [],[]
#     once = 1

#     ani = animation.FuncAnimation(fig, update, 99, fargs=[],
#                                 interval=25, blit=False, repeat = False)  # interval/ms, blit = False/without return
#     # ani.save('./results_cmp/bnet/video' + format(iter, '02d') +'.mp4')
#     ani.save('./results_cmp/inv_10000/video.mp4')
#     plt.show()

#     safety = np.array(safety)
#     traj = np.array(traj)
#     data = {'traj': traj, 'safe': safety}
#     import pickle
#     output = open('./results_cmp/inv_10000/data' + format(iter, '04d') +'.pkl', 'wb') 
#     pickle.dump(data, output)
#     output.close()
# ###########################batch test end


ani = animation.FuncAnimation(fig, update, 99, fargs=[],
                                interval=25, blit=False, repeat = False)  # interval/ms, blit = False/without return
ani.save('./logs/video.mp4')   # bnet
plt.show()

safety = np.array(safety)
tt = np.linspace(0, 9.9, 100)
plt.figure(2)
plt.plot(tt, safety, 'g-', label = 'invariance')
plt.plot([0, 9.9], [0, 0], 'k--', label = 'safe set boundary')
plt.ylabel('$b(x, x_p)$', fontsize=16)
plt.xlabel('$t/s$', fontsize=16)
plt.legend(fontsize = 14,loc ='upper right')

plt.savefig('./logs/safety_inv.pdf')
plt.figure(3)
plt.plot(tt, control_test.cpu().numpy()[:,0,0], 'r-', tt, control_test.cpu().numpy()[:,0,1], 'g-', tt, pred_u.cpu().numpy()[0:100,0,0], 'r--', tt, pred_u.cpu().numpy()[0:100,0,1], 'g--')
plt.savefig('./logs/ctrl_inv.png')

traj = np.array(traj)
data = {'traj': traj, 'safe': safety}
import pickle
output = open('./logs/data_inv.pkl', 'wb')
pickle.dump(data, output)
output.close()