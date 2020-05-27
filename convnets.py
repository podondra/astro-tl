import copy

import click
import h5py
from sklearn import metrics
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


N_WAVES = 2048


def vgg_net_0():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )


def vgg_net_1():
    return nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(1024 * 16, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )


def vgg_net_2():
    return nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Flatten(),
        nn.Linear(512 * 32, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )


def vgg_net_3():
    return nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 64, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )


def vgg_net_4():
    return nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Flatten(),
        nn.Linear(128 * 128, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )


def vgg_net_5():
    return nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 128, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )


def vgg_net_a():
    return nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 128, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )


def vgg_net_b():
    return nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 128, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )


def vgg_net_d():
    return nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 128, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )


def vgg_net_e():
    return nn.Sequential(
        nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 128, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )


def init_weights(m):
    if type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


def get_convnet(convnet):
    if convnet == "VGG-Net-A":
        return vgg_net_a()
    if convnet == "VGG-Net-B":
        return vgg_net_b()
    if convnet == "VGG-Net-D":
        return vgg_net_d()
    if convnet == "VGG-Net-E":
        return vgg_net_e()
    if convnet == "VGG-Net-0":
        return vgg_net_0()
    if convnet == "VGG-Net-1":
        return vgg_net_1()
    if convnet == "VGG-Net-2":
        return vgg_net_2()
    if convnet == "VGG-Net-3":
        return vgg_net_3()
    if convnet == "VGG-Net-4":
        return vgg_net_4()
    if convnet == "VGG-Net-5":
        return vgg_net_5()


def load_datasets():
    with h5py.File("sdss_dr14.hdf5", "r") as datafile:
        grp = datafile['2048_zwarning==0']
        X = grp["X_tr"][:].reshape(-1, 1, N_WAVES)
        y = grp["y_tr"][:].astype("f4")
        X_va = grp["X_va"][:].reshape(-1, 1, N_WAVES)
        y_va = grp["y_va"][:].astype("f4")
    return TensorDataset(*list(map(torch.from_numpy, [X, y]))), TensorDataset(*list(map(torch.from_numpy, [X_va, y_va])))


def train(convnet, datasets, writer, n_validation):
    dev = torch.device("cuda")
    convnet = convnet.to(dev)

    dl = DataLoader(datasets[0], batch_size=256, shuffle=True)
    X_va = datasets[1].tensors[0].to(dev)
    y_va_dev = datasets[1].tensors[1].to(dev).unsqueeze(-1)
    y_va_cpu = datasets[1].tensors[1]

    opt = optim.Adam(convnet.parameters())
    criterion = nn.BCEWithLogitsLoss()

    i = 0
    best_f1_score = float("inf")
    improvement = True
    while improvement:
        improvement = False
        for xb, yb in dl:
            convnet.train()
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = criterion(convnet(xb), yb.unsqueeze(-1))
            loss.backward()
            opt.step()
            writer.add_scalar("Loss/training", loss.item(), i + 1)

            convnet.eval()
            with torch.no_grad():
                # validation loss and early stopping
                if i % n_validation == 0:
                    outputs = convnet(X_va)
                    loss_va = criterion(outputs, y_va_dev)
                    f1_score = metrics.f1_score(y_va_cpu, torch.sigmoid(outputs).cpu() > 0.5)
                    writer.add_scalar("Loss/validation", loss_va.item(), i + 1)
                    writer.add_scalar("F1 score/validation", f1_score, i + 1)

                    if f1_score < best_f1_score:
                        best_f1_score = f1_score
                        best_state_dict = copy.deepcopy(convnet.state_dict())
                        improvement = True

            # next iteration
            i += 1

    return best_state_dict


CONVNETS = ["VGG-Net-A", "VGG-Net-B", "VGG-Net-D", "VGG-Net-E", "VGG-Net-0",
            "VGG-Net-1", "VGG-Net-2", "VGG-Net-3", "VGG-Net-4", "VGG-Net-5"]

@click.command()
@click.option("--convnet", type=click.Choice(CONVNETS))
@click.option("--n-validation", default=1, show_default=True, help="The number of steps between validation loss evaluations.")
def cli(convnet, n_validation, save_path):
    writer = SummaryWriter(comment="_{}".format(convnet))
    datasets = load_datasets()
    convnet = get_convnet(convnet)
    convnet.apply(init_weights)
    best_state_dict = train(convnet, datasets, writer, n_validation)
    torch.save(best_state_dict, convnet + ".pt")


if __name__ == "__main__":
    cli()
