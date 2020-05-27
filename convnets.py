import copy

import click
import h5py
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
    if convnet == "VGG Net-A":
        return vgg_net_a()
    if convnet == "VGG Net-B":
        return vgg_net_b()
    if convnet == "VGG Net-D":
        return vgg_net_d()
    if convnet == "VGG Net-E":
        return vgg_net_e()
    if convnet == "VGG Net-0":
        return vgg_net_0()
    if convnet == "VGG Net-1":
        return vgg_net_1()
    if convnet == "VGG Net-2":
        return vgg_net_2()
    if convnet == "VGG Net-3":
        return vgg_net_3()
    if convnet == "VGG Net-4":
        return vgg_net_4()
    if convnet == "VGG Net-5":
        return vgg_net_5()


def load_data():
    with h5py.File("sdss_dr14.hdf5", "r") as datafile:
        grp = datafile['2048_zwarning==0']
        X = grp["X_tr"][:]
        y = grp["y_tr"][:]
        X_va = grp["X_va"][:]
        y_va = grp["y_va"][:]

    return {"ds": TensorDataset(*list(map(torch.from_numpy, [X.reshape(-1, 1, N_WAVES), y.astype("f4")]))),
            "X_va": torch.from_numpy(X_va.reshape(-1, 1, N_WAVES)),
            "y_va": torch.from_numpy(y_va.astype("f4"))}


def evaluate(convnet, ds):
    dev = torch.device("cuda")
    dl = DataLoader(ds, batch_size=2 ** 15)
    criterion_sum = nn.BCEWithLogitsLoss(reduction="sum")
    loss_tr = 0
    for xb, yb in dl:
        xb, yb = xb.to(dev), yb.to(dev)
        loss_tr += criterion_sum(convnet(xb), yb.unsqueeze(-1)).item()
    return loss_tr / ds.tensors[0].shape[0]


def train(convnet, data_dict, patiance, n_validation, n_training, writer):
    dev = torch.device("cuda")
    convnet = convnet.to(dev)

    ds, X_va, y_va = data_dict["ds"], data_dict["X_va"].to(dev), data_dict["y_va"].to(dev)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    opt = optim.Adam(convnet.parameters())
    criterion = nn.BCEWithLogitsLoss()

    best_va_loss, j, i = float("inf"), 0, 0
    while j < patiance:
        for xb, yb in dl:
            convnet.train()
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = criterion(convnet(xb), yb.unsqueeze(-1))
            loss.backward()
            opt.step()

            convnet.eval()
            with torch.no_grad():
                # validation loss
                if i % n_validation == 0:
                    loss_va = criterion(convnet(X_va), y_va.unsqueeze(-1)).item()
                    writer.add_scalars("Loss", {"Validation set loss": loss_va}, i + 1)

                    if loss_va < best_va_loss:
                        j = 0
                        best_state_dict = copy.deepcopy(convnet.state_dict())
                        best_va_loss = loss_va
                    else:
                        j += 1

                    if not j < patiance:
                        # last training loss evaluation
                        loss_tr = evaluate(convnet, ds)
                        writer.add_scalars("Loss", {"Training set loss": loss_tr}, i + 1)
                        break

                # training loss
                if i % n_training == 0:
                    loss_tr = evaluate(convnet, ds)
                    writer.add_scalars("Loss", {"Training set loss": loss_tr}, i + 1)

            # next iteration
            i += 1

    return best_state_dict


CONVNETS = ["VGG Net-A",
            "VGG Net-B",
            "VGG Net-D",
            "VGG Net-E",
            "VGG Net-0",
            "VGG Net-1",
            "VGG Net-2",
            "VGG Net-3",
            "VGG Net-4",
            "VGG Net-5"]

@click.command()
@click.option("--convnet",
              type=click.Choice(CONVNETS))
@click.option("-p", "--patiance", default=1024, show_default=True,
              help="The number of times to observe worsenning validation set error before giving up.")
@click.option("--n-validation", default=1, show_default=True,
              help="The number of steps between validation loss evaluations.")
@click.option("--n-training", default=256, show_default=True,
              help="The number of steps between training loss evaluations.")
@click.argument("save_path", type=click.Path(dir_okay=False, writable=True))
def cli(convnet, patiance, n_validation, n_training, save_path):
    writer = SummaryWriter(comment="_{}".format(convnet))
    click.echo("Loading data...")
    data_dict = load_data()
    convnet = get_convnet(convnet)
    convnet.apply(init_weights)
    click.echo("Training...")
    best_state_dict = train(convnet, data_dict, patiance, n_validation, n_training, writer)
    click.echo("Saving...")
    torch.save(best_state_dict, save_path)


if __name__ == "__main__":
    cli()
