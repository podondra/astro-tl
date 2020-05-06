from astropy.io import fits
import h5py
import numpy as np
from sklearn.preprocessing import minmax_scale
from spectres import spectres


ID_DTYPE = [("plate", np.int32), ("mjd", np.int32), ("fiberid", np.int32)]


# get labels
with fits.open("data/DR14Q_v4_4.fits") as qso_hdul:
    qsos = qso_hdul[1].data

    qso_plate_duplicates = qsos["plate_duplicate"][qsos["plate_duplicate"] > 0]
    qso_mjd_duplicates = qsos["mjd_duplicate"][qsos["mjd_duplicate"] > 0]
    qso_fiberid_duplicates = qsos["fiberid_duplicate"][qsos["fiberid_duplicate"] > 0]

    qso_plates = np.concatenate((qsos["plate"], qso_plate_duplicates))
    qso_mjds = np.concatenate((qsos["mjd"], qso_mjd_duplicates))
    qso_fiberids = np.concatenate((qsos["fiberid"], qso_fiberid_duplicates))

    qso_ids = np.zeros(len(qso_plates), dtype=ID_DTYPE)
    qso_ids["plate"], qso_ids["mjd"], qso_ids["fiberid"] = qso_plates, qso_mjds, qso_fiberids

with h5py.File("sdss_dr14.hdf5", "r+") as datafile:
    sdss_dr14 = datafile["sdss_dr14"]

    # filter based on ZWARNING (maybe S/N, Z)
    zwarnings = sdss_dr14["zwarning"][:]
    idx_filter = zwarnings == 0

    # load into memory and then index
    X = sdss_dr14["X"][:][idx_filter]
    fiberids = sdss_dr14["fiberid"][:][idx_filter]
    mjds = sdss_dr14["mjd"][:][idx_filter]
    plates = sdss_dr14["plate"][:][idx_filter]
    sns = sdss_dr14["sn_median_all"][:][idx_filter]
    zs = sdss_dr14["z"][:][idx_filter]
    zwarnings = zwarnings[idx_filter]

    # construct identifier
    n = len(X)
    ids = np.zeros(n, dtype=ID_DTYPE)
    ids["plate"], ids["mjd"], ids["fiberid"] = plates, mjds, fiberids

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

    # split into training, validation and test set
    # sizes according to ILSVRC
    N_VAL, N_TEST = 50000, 100000
    n_tr = n - N_VAL - N_TEST
    # seed from random.org
    rng = np.random.default_rng(seed=55)
    rnd_idx = rng.permutation(n)
    idx_tr, idx_va, idx_te = rnd_idx[:n_tr], rnd_idx[n_tr:n_tr + N_VAL], rnd_idx[n_tr + N_VAL:]

    grp = datafile.create_group("{}_zwarning==0".format(N_WAVES))
    for idx, name in ((idx_tr, "tr"), (idx_va, "va"), (idx_te, "te")):
        grp.create_dataset("X_" + name, data=X[idx])
        grp.create_dataset("y_" + name, data=y[idx])
        grp.create_dataset("id_" + name, data=ids[idx])
        grp.create_dataset("sn_median_all_" + name, data=sns[idx])
        grp.create_dataset("z_" + name, data=zs[idx])
        grp.create_dataset("zwarning_" + name, data=zwarnings[idx])
