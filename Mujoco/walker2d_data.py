# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data

class Walker2dImitationData:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        all_files = sorted(
            [
                os.path.join("data/walker", d)
                for d in os.listdir("data/walker")
                if d.endswith(".npy")
            ]
        )

        self.rng = np.random.RandomState(891374)
        np.random.RandomState(125487).shuffle(all_files)
        # 15% test set, 10% validation set, the rest is for training
        test_n = int(0.15 * len(all_files))
        valid_n = int((0.15 + 0.1) * len(all_files))
        test_files = all_files[:test_n]
        valid_files = all_files[test_n:valid_n]
        train_files = all_files[valid_n:]

        train_x, train_t, train_y = self._load_files(train_files)
        valid_x, valid_t, valid_y = self._load_files(valid_files)
        test_x, test_t, test_y = self._load_files(test_files)

        train_x, train_t, train_y = self.perturb_sequences(train_x, train_t, train_y)
        valid_x, valid_t, valid_y = self.perturb_sequences(valid_x, valid_t, valid_y)
        test_x, test_t, test_y = self.perturb_sequences(test_x, test_t, test_y)

        self.train_x, self.train_times, self.train_y = self.align_sequences(
            train_x, train_t, train_y
        )
        self.valid_x, self.valid_times, self.valid_y = self.align_sequences(
            valid_x, valid_t, valid_y
        )
        self.test_x, self.test_times, self.test_y = self.align_sequences(
            test_x, test_t, test_y
        )
        self.input_size = self.train_x.shape[-1]

        # print("train_times: ", str(self.train_times.shape))
        # print("train_x: ", str(self.train_x.shape))
        # print("train_y: ", str(self.train_y.shape))

        maxs = self.train_x.max(axis=0).max(axis=0)
        mins = self.train_x.min(axis=0).min(axis=0)
        print("Constraints:")
        for i in range(self.input_size):
            print(f"Feature[{i}]: {mins[i]} <= x[{i}] <= {maxs[i]}")

    def align_sequences(self, set_x, set_t, set_y):

        times = []
        x = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_t = set_t[i]
            seq_y = set_y[i]

            for t in range(0, seq_y.shape[0] - self.seq_len, self.seq_len // 4):
                x.append(seq_x[t : t + self.seq_len])
                times.append(seq_t[t : t + self.seq_len])
                y.append(seq_y[t : t + self.seq_len])

        return (
            np.stack(x, axis=0),
            np.expand_dims(np.stack(times, axis=0), axis=-1),
            np.stack(y, axis=0),
        )

    def perturb_sequences(self, set_x, set_t, set_y):

        x = []
        times = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_y = set_y[i]

            new_x, new_times = [], []
            new_y = []

            skip = 0
            for t in range(seq_y.shape[0]):
                skip += 1
                if self.rng.rand() < 0.9:
                    new_x.append(seq_x[t])
                    new_times.append(skip)
                    new_y.append(seq_y[t])
                    skip = 0

            x.append(np.stack(new_x, axis=0))
            times.append(np.stack(new_times, axis=0))
            y.append(np.stack(new_y, axis=0))

        return x, times, y

    def _load_files(self, files):
        all_x = []
        all_t = []
        all_y = []
        for f in files:

            arr = np.load(f)
            x_state = arr[:-1, :].astype(np.float32)
            y = arr[1:, :].astype(np.float32)

            x_times = np.ones(x_state.shape[0])
            all_x.append(x_state)
            all_t.append(x_times)
            all_y.append(y)

            # print("Loaded file '{}' of length {:d}".format(f, x_state.shape[0]))
        return all_x, all_t, all_y


def load_dataset(seq_len):
    dataset = Walker2dImitationData(seq_len)

    train_x = torch.Tensor(dataset.train_x)
    train_y = torch.Tensor(dataset.train_y)
    train_ts = torch.Tensor(np.cumsum(dataset.train_times,axis=1))
    test_x = torch.Tensor(dataset.test_x)
    test_y = torch.LongTensor(dataset.test_y)
    test_ts = torch.Tensor(np.cumsum(dataset.test_times,axis=1))
    # train = data.TensorDataset(train_x, train_ts, train_y)
    # test = data.TensorDataset(test_x, test_ts, test_y)
    
    def select(data, idx):
        return data[idx:idx+100]
    idx = 1  #0
    train_x, train_y = select(train_x, idx), select(train_y, idx)

    train = data.TensorDataset(train_x, train_y)
    test = data.TensorDataset(test_x, test_y)
    trainloader = data.DataLoader(train, batch_size=64, shuffle=False, num_workers=12)
    testloader = data.DataLoader(test, batch_size=64, shuffle=False, num_workers=12)
    in_features = train_x.size(-1)
    return trainloader, testloader, in_features

def get_test_init_states():
    dataset = Walker2dImitationData(1)
    test_x = torch.Tensor(dataset.test_x[:,0,:])
    return test_x

if __name__ == "__main__":
    trainloader, testloader, in_features =  load_dataset(seq_len=15)

    print("Dataset")
    for x,y in testloader:
        print("x.size():",x.size())
        # print("y.size():",t.size())
        print("y.size():",y.size())
        import pdb; pdb.set_trace()
        break
