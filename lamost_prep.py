from astropy.io import fits
import h5py
import numpy as np
from sklearn.preprocessing import minmax_scale
from spectres import spectres


# size according to ILSVRC
N_TEST = 100000
# ID_DTYPE
ID_DTYPE = [("lmjd", np.int32), ("planid", "S40"), ("spid", np.uint8), ("fiberid", np.uint8)]


with fits.open("data/lamost_phase1_v02.fits") as qso_hdul:
    qsos = qso_hdul[1].data

    qso_ids = np.zeros(len(qsos), dtype=ID_DTYPE)
    qso_ids["lmjd"] = qsos["mjd"] + 1
    qso_ids["planid"] = np.char.strip(qsos["planid"])
    qso_ids["spid"] = qsos["spid"]
    qso_ids["fiberid"] = qsos["fiberid"]

with h5py.File("lamost_dr5.hdf5", "r+") as datafile:
    lamost_dr5 = datafile["lamost_dr5"]

    # no filtering so far

    # load all into memory and then index
    X = lamost_dr5["X"][:]
    lmjds = lamost_dr5["lmjd"][:]
    planids = lamost_dr5["planid"][:]
    spids = lamost_dr5["spid"][:]
    fiberids = lamost_dr5["fiberid"][:]
    zs = lamost_dr5["z"][:]
    sns = lamost_dr5["snr_ugriz"][:]

    # construct identifier
    n = len(X)
    ids = np.zeros(n, dtype=ID_DTYPE)
    ids["lmjd"], ids["planid"], ids["spid"], ids["fiberid"] = lmjds, planids, spids, fiberids

    # add labels
    y = np.isin(ids, qso_ids, assume_unique=True)

    # resample
    N_WAVES = 2048
    LOGLAMMIN, LOGLAMMAX = 3.5843, 3.9501
    EPS = 0.00005
    # original and new wavelengths (EPS not to get NaNs)
    lam = np.logspace(LOGLAMMIN, LOGLAMMAX, 3659)
    new_lam = np.logspace(LOGLAMMIN + EPS, LOGLAMMAX - EPS, N_WAVES)
    X = spectres(new_lam, lam, X, verbose=True)

    # minmax scale each spectrum
    X = minmax_scale(X, feature_range=(-1, 1), axis=1, copy=False)
    
    # split into training, validation and test set (sizes according to ILSVRC)
    n_tr = n - N_TEST
    # seed from random.org
    rng = np.random.default_rng(seed=26)
    rnd_idx = rng.permutation(n)
    idx_tr, idx_te = rnd_idx[:n_tr], rnd_idx[n_tr:]

    grp = datafile.create_group("{}_nofilter".format(N_WAVES))
    for idx, name in ((idx_tr, "tr"), (idx_te, "te")):
        grp.create_dataset("X_" + name, data=X[idx])
        grp.create_dataset("y_" + name, data=y[idx])
        grp.create_dataset("id_" + name, data=ids[idx])
        grp.create_dataset("z_" + name, data=zs[idx])
        grp.create_dataset("snr_ugriz_" + name, data=sns[idx])
