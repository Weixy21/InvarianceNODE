import numpy as np
import os
import argparse
import torch
import torch.utils.data as data


def cut_in_sequences(x, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len - 1, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(x[start + 1 : end + 1])

    return sequences_x, sequences_y


class CheetahData:
    def __init__(self, seq_len=20):
        all_files = sorted(
            [
                os.path.join("data/cheetah", d)
                for d in os.listdir("data/cheetah")
                if d.endswith(".npy")
            ]
        )

        train_files = all_files[15:25]
        test_files = all_files[5:15]
        valid_files = all_files[:5]

        self.seq_len = seq_len
        self.obs_size = 17

        self.train_x, self.train_y = self._load_files(train_files)
        self.test_x, self.test_y = self._load_files(test_files)
        self.valid_x, self.valid_y = self._load_files(valid_files)

        # print("train_x.shape:", str(self.train_x.shape))
        # print("train_y.shape:", str(self.train_y.shape))
        # print("valid_x.shape:", str(self.valid_x.shape))
        # print("valid_y.shape:", str(self.valid_y.shape))
        # print("test_x.shape:", str(self.test_x.shape))
        # print("test_y.shape:", str(self.test_y.shape))

        # need to transpose the data
        self.train_x = np.transpose(self.train_x, [1, 0, 2])
        self.train_y = np.transpose(self.train_y, [1, 0, 2])
        self.test_x = np.transpose(self.test_x, [1, 0, 2])
        self.test_y = np.transpose(self.test_y, [1, 0, 2])
        self.valid_x = np.transpose(self.valid_x, [1, 0, 2])
        self.valid_y = np.transpose(self.valid_y, [1, 0, 2])

        # maxs = self.train_x.max(axis=0).max(axis=0)
        # mins = self.train_x.min(axis=0).min(axis=0)
        # print("Constraints:")
        # for i in range(self.obs_size):
        #     print(f"Feature[{i}]: {mins[i]} <= x[{i}] <= {maxs[i]}")

    def _load_files(self, files):
        all_x = []
        all_y = []
        for f in files:

            arr = np.load(f)
            arr = arr.astype(np.float32)
            x, y = cut_in_sequences(arr, self.seq_len, 10)

            all_x.extend(x)
            all_y.extend(y)

        return np.stack(all_x, axis=1), np.stack(all_y, axis=1)


def load_dataset(seq_len):
    dataset = CheetahData(seq_len)

    train_x = torch.Tensor(dataset.train_x)
    train_y = torch.Tensor(dataset.train_y)
    test_x = torch.Tensor(dataset.test_x)
    test_y = torch.LongTensor(dataset.test_y)
    train = data.TensorDataset(train_x, train_y)
    test = data.TensorDataset(test_x, test_y)
    trainloader = data.DataLoader(train, batch_size=64, shuffle=True, num_workers=4)
    testloader = data.DataLoader(test, batch_size=64, shuffle=False, num_workers=4)
    in_features = train_x.size(-1)
    return trainloader, testloader, in_features


def get_test_init_states():
    dataset = CheetahData(1)
    test_x = torch.Tensor(dataset.test_x[:, 0, :])
    return test_x


if __name__ == "__main__":
    trainloader, testloader, in_features = load_dataset(seq_len=15)

    print("Dataset")
    for x, y in trainloader:
        print("x.size():", x.size())
        print("y.size():", y.size())
        break