from math import ceil
from time import time

from astropy.io import fits
import h5py
import numpy as np
from mpi4py import MPI


N_WAVES = 3659
LAMMIN, LAMMAX = 3839.7244, 8914.597
LOGLAMMIN, LOGLAMMAX = 3.5843, 3.9501
TEMPLATE = "data/sdss_dr14/{:04d}/spec-{:04d}-{:05d}-{:04d}.fits"

start_t = time()

comm = MPI.COMM_WORLD     # communicator which links all our processes together
rank = comm.Get_rank()    # number which identifies this process
size = comm.Get_size()    # number of processes in a communicator

with fits.open("data/specObj-dr14.fits") as hdulist:
    catalog = hdulist[1].data

# filter based on LAMMIN and LAMMAX
idx = (catalog["WAVEMIN"] <= LAMMIN) & (LAMMAX <= catalog["WAVEMAX"])
n = np.sum(idx)

# metadata to extract
plates = catalog["PLATE"][idx]
mjds = catalog["MJD"][idx]
fiberids = catalog["FIBERID"][idx]
zs = catalog["Z"][idx]
zwarnings = catalog["ZWARNING"][idx]
sn_median_alls = catalog["SN_MEDIAN_ALL"][idx]

# store labels
with fits.open("data/DR14Q_v4_4.fits") as qso_hdul:
    qsos = qso_hdul[1].data
    qso_plates = np.concatenate((qsos["plate"], qsos["plate_duplicate"][qsos["plate_duplicate"] > 0]))
    qso_mjds = np.concatenate((qsos["mjd"], qsos["mjd_duplicate"][qsos["mjd_duplicate"] > 0]))
    qso_fiberids = np.concatenate((qsos["fiberid"], qsos["fiberid_duplicate"][qsos["fiberid_duplicate"] > 0]))

# mode 'x' (create file, fail if exists)
datafile = h5py.File("data.hdf5", 'x', driver="mpio", comm=comm)
grp = datafile.create_group("sdss_dr14")

X_dset = grp.create_dataset("X", shape=(n, N_WAVES), dtype=np.float32)
y_dset = grp.create_dataset("y", shape=(n,), dtype=np.bool, fillvalue=False)
plate_dset = grp.create_dataset("plate", shape=(n,), dtype=np.int32)
mjd_dset = grp.create_dataset("mjd", shape=(n,), dtype=np.int32)
fiberid_dset = grp.create_dataset("fiberid", shape=(n,), dtype=np.int32)
z_dset = grp.create_dataset("z", shape=(n,), dtype=np.float32)
zwarning_dset = grp.create_dataset("zwarning", shape=(n,), dtype=np.int32)
sn_median_all_dset = grp.create_dataset("sn_median_all", shape=(n,), dtype=np.float32)

# divide data between processes
chunk = ceil(n / size)
start = rank * chunk
end = start + chunk if start + chunk <= n else n

for i in range(start, end):
    plate, mjd, fiberid = plates[i], mjds[i], fiberids[i]
    filepath = TEMPLATE.format(plate, plate, mjd, fiberid)
    with fits.open(filepath) as hdulist:
        data = hdulist[1].data
        loglam = data["loglam"]
        flux = data["flux"][(LOGLAMMIN <= loglam) & (loglam <= LOGLAMMAX)]
        # store in datasets (flux, plate, mjd, fiberid, ZWARNINGS, S/N...)
        X_dset[i] = flux
        y_dset[i] = np.any((qso_plates == plate) & (qso_mjds == mjd) & (qso_fiberids == fiberid))
        plate_dset[i] = plate
        mjd_dset[i] = mjd
        fiberid_dset[i] = fiberid
        z_dset[i] = zs[i]
        zwarning_dset[i] = zwarnings[i]
        sn_median_all_dset[i] = sn_median_alls[i]

datafile.close()

print("process {} ({}): {}".format(rank, size, time() - start_t))
