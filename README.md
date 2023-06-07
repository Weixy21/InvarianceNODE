# Forward invariance of neural ODEs

Causal manipulation of neural ODEs (via model parameters or external inputs) to achieve performance guarantees, such as safety 

![pipeline](imgs/obs_walker_invariance.mp4) 

There are four simple modelling demos using neural ODEs with performance specifications (spiral curve regression, convexity portrait, Mujoco, and end-to-end lidar-based autonomous driving).

## Setup

    ```
    $ conda create -n invODE python=3.8
    $ conda activate invODE
    $ pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    $ pip install pytorch-lightning==1.5.8 opencv-python==4.5.2.54 matplotlib==3.5.1 ffio==0.1.0  descartes==1.1.0  pyrender==0.1.45  pandas==1.3.5 shapely==1.7.1 scikit-video==1.1.11 scipy==1.6.3 h5py==3.1.0
    $ pip install qpth cvxpy cvxopt
    $ pip install torchdiffeq
    ```


If you find this helpful, please cite our work:
```
@inproceedings{xiao2023inv,
  title = {On the Forward Invariance of Neural ODEs},
  author = {Wei Xiao and Tsun-Hsuan Wang and Ramin Hasani and Mathias Lechner and Yutong Ban and Chuang Gan and Daniela Rus},
  booktitle = {International Conference on Machine Learning},
  year = {2023}
}
```