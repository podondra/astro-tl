# Transfer Leaning in Astronomical Spectroscopy

Use of transfer learning (in the context of deep learning) to classify QSOs in LAMOST (target domain) using SDSS (source domain) as the training data.

## Envrinment Setup

The envrinment is based on Python 3.7.2, CUDA 10.1 and Parallel HDF5.

    $ module load Python/3.7.2-fosscuda-2019a
    $ module load HDF5/1.10.5-gompic-2019a
    $ virtualenv venv    # create a virtual environment
    $ source venv/bin/activate    # activate the environment
    $ # install h5py with MPI mode enabled
    $ pip install mpi4py
    $ CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/parallel-hdf5 pip install --no-binary=h5py h5py
    $ pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    $ pip install -r requirements.txt
