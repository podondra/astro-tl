import h5py
import torch
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

import convnets


N_WAVES = 2048


if __name__ == "__main__":
    with h5py.File("lamost_dr5.hdf5", "r") as datafile:
        grp = datafile["2048_nofilter"]
        X = grp["X_tr"][:].reshape(-1, 1, N_WAVES)
        y = grp["y_tr"][:].astype("f4")
        X_va = grp["X_va"][:].reshape(-1, 1, N_WAVES)
        y_va = grp["y_va"][:].astype("f4")
    datasets = TensorDataset(*list(map(torch.from_numpy, [X, y]))), TensorDataset(*list(map(torch.from_numpy, [X_va, y_va])))

    model = convnets.vgg_net_a()
    model.load_state_dict(torch.load("VGG-Net-A.pt"))
    writer = SummaryWriter(comment="_transfer")
    best_state_dict = convnets.train(model, datasets, writer, n_validation=1)
    torch.save(best_state_dict, "transfer.pt")
