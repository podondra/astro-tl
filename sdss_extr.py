from math import ceil

from astropy.io import fits
import h5py
import numpy as np
from mpi4py import MPI


N_WAVES = 3659
LAMMIN, LAMMAX = 3839.7244, 8914.597
LOGLAMMIN, LOGLAMMAX = 3.5843, 3.9501

BASE = "/home/podszond/qso/"
TEMPLATE = "data/sdss_dr14/{:04d}/spec-{:04d}-{:05d}-{:04d}.fits"


comm = MPI.COMM_WORLD     # communicator which links all our processes together
rank = comm.Get_rank()    # number which identifies this process
size = comm.Get_size()    # number of processes in a communicator

with fits.open(BASE + "data/specObj-dr14.fits") as hdulist:
    catalog = hdulist[1].data

# filter based on LAMMIN and LAMMAX
idx = (catalog["WAVEMIN"] <= LAMMIN) & (LAMMAX <= catalog["WAVEMAX"])

# metadata to extract
plates = catalog["PLATE"][idx]
mjds = catalog["MJD"][idx]
fiberids = catalog["FIBERID"][idx]
zs = catalog["Z"][idx]
zwarnings = catalog["ZWARNING"][idx]
sns = catalog["SN_MEDIAN_ALL"][idx]

# divide data between processes
n = np.sum(idx)
chunk = ceil(n / size)
start = rank * chunk
end = start + chunk if start + chunk <= n else n

with h5py.File("data.hdf5", 'a', driver="mpio", comm=comm) as datafile:
    grp = datafile.create_group("sdss_dr14")
    
    X_dset        = grp.create_dataset("X", shape=(n, N_WAVES), dtype=np.float32)
    plate_dset    = grp.create_dataset("plate", data=plates, dtype=np.int32)
    mjd_dset      = grp.create_dataset("mjd", data=mjds, dtype=np.int32)
    fiberid_dset  = grp.create_dataset("fiberid", data=fiberids, dtype=np.int32)
    z_dset        = grp.create_dataset("z", data=zs, dtype=np.float32)
    zwarning_dset = grp.create_dataset("zwarning", data=zwarnings, dtype=np.int32)
    sn_dset       = grp.create_dataset("sn_median_all", data=sns, dtype=np.float32)

    for i in range(start, end):
        plate, mjd, fiberid = plates[i], mjds[i], fiberids[i]
        filepath = TEMPLATE.format(plate, plate, mjd, fiberid)
        with fits.open(BASE + filepath) as hdulist:
            data = hdulist[1].data
            loglam = data["loglam"]
            flux = data["flux"][(LOGLAMMIN <= loglam) & (loglam <= LOGLAMMAX)]
            X_dset[i] = flux
