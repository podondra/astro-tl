from math import ceil

from astropy.io import fits
from astropy.time import Time
import h5py
import numpy as np
from mpi4py import MPI


N_WAVES = 3659
LOGLAMMIN, LOGLAMMAX = 3.5843, 3.9501
EPS = 0.00005

BASE = "/home/podszond/qso/"
TEMPLATE = "data/lamost_dr5_v3/{}/{}/spec-{}-{}_sp{:02d}-{:03d}.fits.gz"


comm = MPI.COMM_WORLD     # communicator which links all our processes together
rank = comm.Get_rank()    # number which identifies this process
size = comm.Get_size()    # number of processes in a communicator

# open LAMOST catalog
with fits.open(BASE + "data/dr5_v3.fits") as hdulist:
    catalog = hdulist[1].data

# metadata to extract
lmjds = catalog["lmjd"]
planids = catalog["planid"]
spids = catalog["spid"]
fiberids = catalog["fiberid"]
zs = catalog["z"]
snrs = np.stack([catalog["snru"], catalog["snrg"], catalog["snrr"], catalog["snri"], catalog["snrz"]], axis=1)

# divide data between processes
n = len(catalog)
chunk = ceil(n / size)
start = rank * chunk
end = start + chunk if start + chunk <= n else n

with h5py.File("data.hdf5", 'a', driver="mpio", comm=comm) as datafile:
    grp = datafile.create_group("lamost_dr5")

    X_dset       = grp.create_dataset("X", shape=(n, N_WAVES), dtype=np.float32)
    lmjd_dset    = grp.create_dataset("lmjd", data=lmjds, dtype=np.int32)
    planid_dset  = grp.create_dataset("planid", data=planids.astype("S40"), dtype="S40")
    spid_dset    = grp.create_dataset("spid", data=spids, dtype=np.uint8)
    fiberid_dset = grp.create_dataset("fiberid", data=fiberids, dtype=np.uint8)
    z_dset       = grp.create_dataset("z", data=zs, dtype=np.float32)
    snr_dset     = grp.create_dataset("snr_ugriz", data=snrs, dtype=np.float32)

    for i in range(start, end):
        lmjd, planid, spid, fiberid = lmjds[i], planids[i], spids[i], fiberids[i]
        obsdate = Time(lmjd - 1, format="mjd").datetime.strftime("%Y%m%d")
        filepath = TEMPLATE.format(obsdate, planid, lmjd, planid, spid, fiberid)
        with fits.open(BASE + filepath) as hdulist:
            data = hdulist[0].data
            loglam = np.log10(data[2])
            X_dset[i] = data[0][(LOGLAMMIN - EPS <= loglam) & (loglam <= LOGLAMMAX + EPS)]
