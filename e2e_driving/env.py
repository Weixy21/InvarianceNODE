import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def map(veh_num, road_num):
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.axis('equal')

    veh, veh_un, speed_handle = [], [], []
    for i in range(veh_num):
        if i == 0:
            line, = ax.plot([], [], color='r', linewidth=3.0)
        else:
            line, = ax.plot([], [], color='k', linewidth=3.0)
        veh.append(line)
        if i == 0:
            line, = ax.plot([], [], color='r', linestyle=':', marker='o')
        else:
            line, = ax.plot([], [], color='k', linestyle=':')
        veh_un.append(line)
        text_handle = ax.text([],[], '', fontsize=12)
        speed_handle.append(text_handle)
    pred_handle, = ax.plot([], [], color='r', linestyle='--')
    
    ego_correct, = ax.plot([], [], color='g', linestyle=':', marker='o')

    road = []
    for i in range(road_num + 1): 
        if i == 0 or i == road_num:
            line, = ax.plot([], [], color='k', linewidth=4.0)
        else:
            line, = ax.plot([], [], color='k', linestyle=':')
        road.append(line)
    line, = ax.plot([], [], color='m', marker='^', linewidth=60) # one markers
    road.append(line)
    line, = ax.plot([], [], color='m', marker='v', linewidth=60) # one markers
    road.append(line)
    # speed_handle = ax.text([],[], '')
    plt.show(block = False)
    plt.pause(0.02)
    return ax, fig, veh, veh_un, road, speed_handle, pred_handle, ego_correct

def search_the_last(veh_state, active_num, current_id):
    veh_last_pos = veh_state[0,0]
    for i in range(active_num):
        if int(veh_state[i,5]) == current_id:
            veh_last_pos = max(veh_last_pos, veh_state[i,0])
    
    return veh_last_pos

def search_preceding(veh_state, active_num, veh_pos, lane_id):
    veh_pred_pos = veh_pos + 40
    veh_pred_vel = None
    for i in range(active_num):
        if int(veh_state[i,5]) == lane_id:
            if veh_state[i,0] > veh_pos and veh_state[i,0] < veh_pred_pos:
                veh_pred_pos = veh_state[i,0]
                veh_pred_vel = veh_state[i,3]
    
    return veh_pred_pos, veh_pred_vel

def search_vehicles(veh_state, active_num):
    corner = np.array([[-2.5, -1.0], [-2.5, +1.0], [+2.5, +1.0], [+2.5, -1.0]])
    indexes = []
    obs = []
    if(active_num > 1):
        for i in range(1, active_num, 1):
            foot = veh_state[i,0:2] + corner
            within = False
            for j in range(4):
                if(foot[j,0] - veh_state[0,0])**2 + (foot[j,1] - veh_state[0,1])**2 <= 20**2:
                    within = True
                    break
            if within == True:
                indexes.append(i)
                obs.append([veh_state[i,0], veh_state[i,1], veh_state[i,3], veh_state[i,5]])
    return indexes, obs

def sensing(veh_state, indexes, with_noise):
    num = len(indexes)
    results = []
    sensing_range = 20
    for theta in np.arange(0, 2*np.pi, 2*np.pi/100.):
        if theta >= 0 and theta < np.pi/2:
            if np.abs(theta) < 0.001:
                sx, sy = sensing_range, 0
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        if rx > 2.5 and ry >= -1 and ry <= 1:
                            sx = min(rx - 2.5, sx)
                            sy = 0
            else:
                sx, sy = sensing_range*np.cos(theta), sensing_range*np.sin(theta)
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        temp = np.tan(theta)*(rx - 2.5)
                        if rx > 2.5 and temp >= ry - 1 and temp <= ry + 1:
                            if rx - 2.5 < sx:
                                sx = rx - 2.5
                                sy = temp
                        temp = (ry-1)/np.tan(theta)
                        if ry > 1 and temp >= rx - 2.5 and temp <= rx + 2.5:
                            if ry - 1 < sy:
                                sx = temp
                                sy = ry - 1
        elif theta >= np.pi/2 and theta < np.pi:
            if np.abs(theta - np.pi/2) < 0.001:
                sx, sy = 0, sensing_range
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        if ry > 1 and rx >= -2.5 and rx <= 2.5:
                            sx = 0
                            sy = min(ry - 1, sy)
            else:
                sx, sy = sensing_range*np.cos(theta), sensing_range*np.sin(theta)
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        temp = np.tan(theta)*(rx + 2.5)
                        if rx < -2.5 and temp >= ry - 1 and temp <= ry + 1:
                            if rx + 2.5 > sx:
                                sx = rx + 2.5
                                sy = temp
                        temp = (ry-1)/np.tan(theta)
                        if ry > 1 and temp >= rx - 2.5 and temp <= rx + 2.5:
                            if ry - 1 < sy:
                                sx = temp
                                sy = ry - 1
        elif theta >= np.pi and theta < 3*np.pi/2:
            if np.abs(theta - np.pi) < 0.001:
                sx, sy = -sensing_range, 0
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        if rx < -2.5 and ry >= -1 and ry <= 1:
                            sx = max(rx + 2.5, sx)
                            sy = 0
            else:
                sx, sy = sensing_range*np.cos(theta), sensing_range*np.sin(theta)
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        temp = np.tan(theta)*(rx + 2.5)
                        if rx < -2.5 and temp >= ry - 1 and temp <= ry + 1:
                            if rx + 2.5 > sx:
                                sx = rx + 2.5
                                sy = temp
                        temp = (ry+1)/np.tan(theta)
                        if ry < -1 and temp >= rx - 2.5 and temp <= rx + 2.5:
                            if ry + 1 > sy:
                                sx = temp
                                sy = ry + 1
        else:
            if np.abs(theta - 3*np.pi/2) < 0.001:
                sx, sy = 0, -sensing_range
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        if ry < -1 and rx >= -2.5 and rx <= 2.5:
                            sx = 0
                            sy = max(ry + 1, sy)
            else:
                sx, sy = sensing_range*np.cos(theta), sensing_range*np.sin(theta)
                if num > 0:
                    for i in range(num):
                        rx, ry = veh_state[indexes[i], 0] - veh_state[0, 0], veh_state[indexes[i], 1] - veh_state[0, 1]
                        temp = np.tan(theta)*(rx - 2.5)
                        if rx > 2.5 and temp >= ry - 1 and temp <= ry + 1:
                            if rx - 2.5 < sx:
                                sx = rx - 2.5
                                sy = temp
                        temp = (ry+1)/np.tan(theta)
                        if ry < -1 and temp >= rx - 2.5 and temp <= rx + 2.5:
                            if ry + 1 > sy:
                                sx = temp
                                sy = ry + 1
        results.append([sx, sy])
    results = np.array(results)

    ################################add noise
    if with_noise:
        for jj in np.arange(100):
            noise = 200*0.02*np.random.rand()
            results[jj, 0] = results[jj, 0] - noise*np.cos(2*np.pi/100*jj)
            results[jj, 1] = results[jj, 1] - noise*np.sin(2*np.pi/100*jj)


    len0 = results.shape[0]
    dist = []
    for i in range(len0):
        temp = np.sqrt((results[i,0])**2 + (results[i,1])**2)
        dist.append(temp)
    dist = np.array(dist)
    return results, dist