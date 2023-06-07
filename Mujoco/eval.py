import argparse
import numpy as np

import torch
import os


parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--seq_len', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument("--dataset", default="walker2d")
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--consider_noise', action='store_true')
parser.add_argument('--hard_truncate', action='store_true')
parser.add_argument('--num_var', type=int, default=1)
parser.add_argument('--study_obs', action='store_true')
args = parser.parse_args()

from train import ODEFunc
# from walker2d_data import load_dataset
import walker2d_data 
import cheetah_data

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
# device = "cpu"


batch_id = 0
time_delta = 0.1

# trainloader, testloader, in_features = load_dataset(seq_len=args.seq_len)


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

func = ODEFunc(in_features).to(device)
func_trunc = ODEFunc(in_features).to(device)


import matplotlib.pyplot as plt
plt.style.use('bmh')
fig = plt.figure(figsize=(8, 4), facecolor='white')
ax_safe = fig.add_subplot(111, frameon=False)


def visualize(torso, hip_joint, knee_left, knee_right, ankle_left, ankle_right, foot_left, foot_right, ii):

    ax_safe.cla()
    ax_safe.set_title('Trajectory')
    ax_safe.set_xlabel('$x/m$')
    ax_safe.set_ylabel('$y/m$')
    ax_safe.plot([torso[0].cpu().numpy(), hip_joint[0].cpu().numpy()], [torso[1].cpu().numpy(), hip_joint[1].cpu().numpy()], 'r-')
    ax_safe.plot([hip_joint[0].cpu().numpy(), knee_left[0].cpu().numpy()], [hip_joint[1].cpu().numpy(), knee_left[1].cpu().numpy()], 'g-')
    ax_safe.plot([knee_left[0].cpu().numpy(), ankle_left[0].cpu().numpy()], [knee_left[1].cpu().numpy(), ankle_left[1].cpu().numpy()], 'g-')
    ax_safe.plot([ankle_left[0].cpu().numpy(), foot_left[0].cpu().numpy()], [ankle_left[1].cpu().numpy(), foot_left[1].cpu().numpy()], 'g-')
    ax_safe.plot([hip_joint[0].cpu().numpy(), knee_right[0].cpu().numpy()], [hip_joint[1].cpu().numpy(), knee_right[1].cpu().numpy()], 'b-')
    ax_safe.plot([knee_right[0].cpu().numpy(), ankle_right[0].cpu().numpy()], [knee_right[1].cpu().numpy(), ankle_right[1].cpu().numpy()], 'b-')
    ax_safe.plot([ankle_right[0].cpu().numpy(), foot_right[0].cpu().numpy()], [ankle_right[1].cpu().numpy(), foot_right[1].cpu().numpy()], 'b-')
    
    ax_safe.axis('equal')
    
    fig.tight_layout()
    
    plt.savefig('./pics/pic{:03d}.png'.format(ii))
    plt.draw()
    plt.pause(0.001)

    plt.show(block=False)



if args.dataset == "walker2d":
    func.load_state_dict(torch.load(f"ckpt/walker2d_ep070.pth"))
    func.eval()
    func_trunc.load_state_dict(torch.load(f"ckpt_tunc/walker2d_ep020.pth"))
    func_trunc.eval()
else:
    func.load_state_dict(torch.load(f"ckpt/halfcheetah_ep140.pth"))
    func.eval()
    func_trunc.load_state_dict(torch.load(f"ckpt_tunc/halfcheetah_ep140.pth"))
    func_trunc.eval()
total = len(testloader) - 1


with torch.no_grad():
    test_losses = []
    test_losses_ode = []
    test_losses_trunc = []
    min_inv, min_ode, min_trunc = [],[],[]
    # for batch_y0, batch_t, batch_y in iter(testloader):
    for batch_y0, batch_y in iter(trainloader):
        batch_y0 = batch_y0.to(device)
        
        ################################################invariance collision avoidance video and figures   - trainloader
        # video generation with invariance
        if args.study_obs:
            func.study_obs = args.study_obs
            batch_y0 = batch_y0[:,0,:]
            batch_t = torch.linspace(
                time_delta, args.seq_len * time_delta, args.seq_len).to(device)
            batch_y = batch_y.to(device)
            
            func.use_dcbf = True
            batch_t = torch.tensor([0., 0.1]).to(device)
            seq = batch_y.shape[1]
            nbat = batch_y.shape[0]
            pred_y = []
            seq = 40  # manually change the total simulation step number

            import gym
            from gym.wrappers import Monitor
            if args.dataset == "walker2d":
                env = gym.make('Walker2d-v2')
            else:
                env = gym.make('HalfCheetah-v2')
            
            env = Monitor(env, './gym', force=True) 
            env.reset()
            if args.dataset == "walker2d":
                j = 51 # pick one for generating video
                y0 = batch_y0[j,:].clone()
                y0[0] -= 0.1   # modify the initial condition
                y0[9] *= 0.6
            else:
                j = 28
                y0 = batch_y0[j,:].clone()
            if args.dataset == "halfcheetah":
                y0t = batch_y[j,:]
            pred_yj = []
            safety = [torch.tensor([1.69 - y0[0] - 0.3*y0[9]])]
            height = []
            pos_x = 0
            for i in range(seq):
                print('batch_id: ', batch_id, '/', total, 'seq: ', j, 'point: ', i)
                func.pos_x = torch.tensor(pos_x).clone().detach()

                pred = odeint(func, y0, batch_t, method = "fixed_adams")   #
                pred_yj.append(pred[-1,:].unsqueeze(0))
                y0 = pred[-1,:]

                ################################ceiling
                # b = 1.69 - pred[-1,0] - 0.3*pred[-1,9]

                ################################ball
                b = (pos_x - 2)**2 + (y0[0]- 1.8)**2 - 0.6**2
                b2 = (pos_x - 3)**2 + (y0[0]- 1.8)**2 - 0.6**2
                b3 = (pos_x - 4)**2 + (y0[0]- 1.8)**2 - 0.6**2
                safety.append(torch.min(torch.tensor([b,b2,b3])).unsqueeze(0))
                height.append(y0[0].unsqueeze(0))

                if args.dataset == "walker2d":
                    pos_x = pos_x + y0[8]*0.01
                else:
                    pos_x = pos_x + y0[8]*0.1
                
                
                qpos = np.concatenate([[pos_x], y0[:8].numpy()])
                qvel = y0[8:].numpy()
                env.set_state(qpos, qvel)
                env.step(np.zeros((env.action_space.shape[0],)))

            env.close()

            pred_yj = torch.cat(pred_yj, dim = 0)
            safety = torch.cat(safety, dim = 0)
            height = torch.cat(height, dim = 0)

            # video generate without invariance
            y0 = batch_y0[j,:].clone()
            y0[0] -= 0.1
            y0[9] *= 0.6
            pred_yj = []
            safety2 = [torch.tensor([1.69 - y0[0] - 0.3*y0[9]])]
            safety2 = []
            func.use_dcbf = False
            pos_x  = 0
            if args.dataset == "walker2d":
                env = gym.make('Walker2d-v2')
            else:
                env = gym.make('HalfCheetah-v2')
            env = Monitor(env, './gym_wo_inv', force=True)
            env.reset()
            for i in range(seq):
                print('batch_id: ', batch_id, '/', total, 'seq: ', j, 'point: ', i)
                pred = odeint(func, y0, batch_t)
                pred_yj.append(pred[-1,:].unsqueeze(0))
                y0 = pred[-1,:]

                ################################ceiling
                # b = 1.69 - pred[-1,0] - 0.3*pred[-1,9]

                ################################ball
                b = (pos_x - 2)**2 + (y0[0]- 1.8)**2 - 0.6**2
                b2 = (pos_x - 3)**2 + (y0[0]- 1.8)**2 - 0.6**2
                b3 = (pos_x - 4)**2 + (y0[0]- 1.8)**2 - 0.6**2
                safety2.append(torch.min(torch.tensor([b,b2,b3])).unsqueeze(0))
                
                pos_x = pos_x + y0[8]*0.01

                qpos = np.concatenate([[pos_x], y0[:8].numpy()])
                qvel = y0[8:].numpy()
                env.set_state(qpos, qvel)
                env.step(np.zeros((env.action_space.shape[0],)))

            env.close()
            pred_yj = torch.cat(pred_yj, dim = 0)
            safety2 = torch.cat(safety2, dim = 0)

            import matplotlib.pyplot as plt
            plt.style.use('bmh')
            fig = plt.figure(figsize=(8, 4), facecolor='white')
            ax_safe = fig.add_subplot(111, frameon=False)

            ax_safe.cla()
            ax_safe.set_title('Safety portrait')
            ax_safe.set_xlabel('time step')
            ax_safe.set_ylabel('$b(x)$')
            ax_safe.plot(safety.cpu().numpy(), 'g-', label = 'neural ODE + invariance')
            ax_safe.plot(safety2.cpu().numpy(), 'r-', label = 'neural ODE')
            ax_safe.plot([0, seq], [0,0], 'k--', label = 'safety boundary')
            ax_safe.set_ylim(-0.4, 1)
            ax_safe.set_xlim(-5, seq)
            ax_safe.legend(loc ='upper right')

            fig.tight_layout()
            
            plt.savefig('safety{:03d}.pdf'.format(2))
            plt.draw()
            plt.pause(0.001)

            plt.show(block=False)


            exit()
        #****************************************************************
        else:
        ################################################invariance - joint limitation test
            batch_y0 = batch_y0[:,0,:]
            batch_t = torch.linspace(
                time_delta, args.seq_len * time_delta, args.seq_len).to(device)
            batch_y = batch_y.to(device)
            
            func.use_dcbf = True
            batch_t = torch.tensor([0., 0.1]).to(device)
            seq = batch_y.shape[1]
            nbat = batch_y.shape[0]
            pred_y = []
            for j in range(nbat):
                y0 = batch_y0[j,:]
                pred_yj = []
                for i in range(seq):
                    print('batch_id: ', batch_id, '/', total, 'seq: ', j, 'point: ', i)
                    pred = odeint(func, y0, batch_t, method = "fixed_adams")
                    pred_yj.append(pred[-1,:].unsqueeze(0))
                    y0 = pred[-1,:]
                pred_yj = torch.cat(pred_yj, dim = 0)
                pred_y.append(pred_yj.unsqueeze(0))
            pred_y = torch.cat(pred_y, dim=0)
        
            loss = torch.mean(torch.abs(pred_y - batch_y)).detach().cpu().item()

            min1 = torch.min(high - pred_y).unsqueeze(0)
            min2 = torch.min(pred_y - low).unsqueeze(0)
            min0 = torch.cat([min1,min2], dim = 0)
            min_inv.append(min0)

            test_losses.append(loss)
            print(f"Invarince Test_loss ={np.mean(test_losses):0.4g}")

            

            ##################################################neural ODE
            batch_t = torch.linspace(
                time_delta, args.seq_len * time_delta, args.seq_len).to(device)
            func.use_dcbf = False

            pred_y = odeint(func, batch_y0, batch_t)
            pred_y = torch.transpose(pred_y, 0, 1)
        
            loss = torch.mean(torch.abs(pred_y - batch_y)).detach().cpu().item()

            min1 = torch.min(high - pred_y).unsqueeze(0)
            min2 = torch.min(pred_y - low).unsqueeze(0)
            min0 = torch.cat([min1,min2], dim = 0)
            min_ode.append(min0)

            test_losses_ode.append(loss)
            print(f"neural ODE Test_loss ={np.mean(test_losses_ode):0.4g}")

            ##################################################hard trunc
            func_trunc.use_dcbf = False
            pred_y = odeint(func_trunc, batch_y0, batch_t)
            pred_y = torch.transpose(pred_y, 0, 1)

            trunc_y = torch.clip(pred_y, low, high)
            loss = torch.mean(torch.abs(trunc_y - batch_y)).detach().cpu().item()

            min1 = torch.min(high - pred_y).unsqueeze(0)
            min2 = torch.min(pred_y - low).unsqueeze(0)
            min0 = torch.cat([min1,min2], dim = 0)
            min_trunc.append(min0)

            test_losses_trunc.append(loss)
            print(f"hard trunc Test_loss ={np.mean(test_losses_trunc):0.4g}")

            batch_id = batch_id + 1
            if(batch_id == 10):
                break

min_inv = torch.cat(min_inv, dim = 0)
min_ode = torch.cat(min_ode, dim = 0)
min_trunc = torch.cat(min_trunc, dim = 0)

print(f"inv safety ={torch.min(min_inv).cpu().item():0.4g}")
print(f"ode safety ={torch.min(min_ode).cpu().item():0.4g}")
print(f"trunc safety ={torch.min(min_trunc).cpu().item():0.4g}")











###########################################################################Batch test

# with torch.no_grad():
#     test_losses = []
#     test_losses_ode = []
#     test_losses_trunc = []
#     # for batch_y0, batch_t, batch_y in iter(testloader):
#     for batch_y0, batch_y in iter(testloader):
#         batch_y0 = batch_y0.to(device)
        
#         ################################################invariance
#         batch_y0 = batch_y0[:,0,:]
#         batch_t = torch.linspace(
#             time_delta, args.seq_len * time_delta, args.seq_len).to(device)
#         batch_y = batch_y.to(device)
        
#         func.use_dcbf = True
#         batch_t = torch.tensor([0., 0.1]).to(device)
#         seq = batch_y.shape[1]
#         nbat = batch_y.shape[0]
#         pred_y = []
#         for j in range(nbat):
#             y0 = batch_y0[j,:]
#             pred_yj = []
#             for i in range(seq):
#                 print('batch_id: ', batch_id, '/', total, 'seq: ', j, 'point: ', i)
#                 pred = odeint(func, y0, batch_t)
#                 pred_yj.append(pred[-1,:].unsqueeze(0))
#                 y0 = pred[-1,:]
#             pred_yj = torch.cat(pred_yj, dim = 0)
#             pred_y.append(pred_yj.unsqueeze(0))
#         pred_y = torch.cat(pred_y, dim=0)
       
#         loss = torch.mean(torch.abs(pred_y - batch_y)).detach().cpu().item()
#         test_losses.append(loss)
#         print(f"Invarince Test_loss ={np.mean(test_losses):0.4g}")

#         ##################################################neural ODE
#         batch_t = torch.linspace(
#             time_delta, args.seq_len * time_delta, args.seq_len).to(device)
#         func.use_dcbf = False

#         pred_y = odeint(func, batch_y0, batch_t)
#         pred_y = torch.transpose(pred_y, 0, 1)
       
#         loss = torch.mean(torch.abs(pred_y - batch_y)).detach().cpu().item()
#         test_losses_ode.append(loss)
#         print(f"neural ODE Test_loss ={np.mean(test_losses_ode):0.4g}")

#         ##################################################hard trunc
#         func_trunc.use_dcbf = False
#         pred_y = odeint(func_trunc, batch_y0, batch_t)
#         pred_y = torch.transpose(pred_y, 0, 1)

#         trunc_y = torch.clip(pred_y, low, high)
#         loss = torch.mean(torch.abs(trunc_y - batch_y)).detach().cpu().item()
#         test_losses_trunc.append(loss)
#         print(f"hard trunc Test_loss ={np.mean(test_losses_trunc):0.4g}")

#         batch_id = batch_id + 1

#     print(f"Invarince Test_loss ={np.mean(test_losses):0.4g}")
#     print(f"Invarince Test_loss_std ={np.std(test_losses):0.4g}")
#     print(f"neural ODE Test_loss ={np.mean(test_losses_ode):0.4g}")
#     print(f"neural ODE Test_loss_std ={np.std(test_losses_ode):0.4g}")
#     print(f"hard trunc Test_loss ={np.mean(test_losses_trunc):0.4g}")
#     print(f"hard trunc Test_loss_std ={np.std(test_losses_trunc):0.4g}")
