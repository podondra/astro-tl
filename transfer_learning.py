import h5py
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

import convnets


N_WAVES = 2048


if __name__ == "__main__":
    with h5py.File("lamost_dr5.hdf5", "r") as datafile:
        grp = datafile["2048_nofilter"]
        X = grp["X_tr"][:]
        y = grp["y_tr"][:]
        X_va = grp["X_va"][:]
        y_va = grp["y_va"][:]

    data_dict = {
        "ds": TensorDataset(*list(map(torch.from_numpy, [X.reshape(-1, 1, N_WAVES), y.astype("f4")]))),
        "X_va": torch.from_numpy(X_va.reshape(-1, 1, N_WAVES)),
        "y_va": torch.from_numpy(y_va.astype("f4"))
    }

    model = convnets.vgg_net_a()
    model.load_state_dict(torch.load("vgg_net-a.pt"))
    writer = SummaryWriter(comment="_transfer-learning")
    best_state_dict = convnets.train(model, data_dict, patiance=1024, n_validation=1, n_training=256, writer=writer)
    torch.save(best_state_dict, "transfer_learning.pt")
