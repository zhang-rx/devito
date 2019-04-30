# coding: utf-8

# Set up inversion parameters.
param = {'t0': 0.,
         'tn': 4000.,              # Simulation last 4 second (4000 ms)
         'f0': 0.008,              # Source peak frequency is 5Hz (0.005 kHz)
         'nshots': 97**2,          # Number of shots to create gradient from
         'm_bounds': (0.08, 0.25), # Set the min and max slowness
         'nbpml': 40,              # nbpml thickness.
         'timestamp':'whatever 3'}

import numpy as np


from devito import Grid

from distributed import Client, LocalCluster, wait

import cloudpickle as pickle

# Import acoustic solver, source and receiver modules.
from examples.seismic import Model, demo_model
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import TimeAxis, PointSource, RickerSource, Receiver

# Import convenience function for plotting results
from examples.seismic import plot_image

from google.cloud import storage

import os
from pathlib import Path

def download_file_from_bucket(filename):
    client = storage.Client(project='seg-demo-project-2')
    bucket = client.get_bucket('datasets-proxy')
    blob = bucket.get_blob(filename)
    with open(filename, 'wb') as f:
        blob.download_to_file(f)

def upload_file_to_bucket(filename):
    client = storage.Client(project='seg-demo-project-2')
    bucket = client.get_bucket('datasets-proxy')
    blob = storage.Blob(filename, bucket)
    with open(filename, 'rb') as f:
        blob.upload_from_file(f)

def from_hdf5(filename, **kwargs):
    import h5py

    f = h5py.File(filename, 'r')

    origin = kwargs.pop('origin', None)
    if origin is None:
        origin_key = kwargs.pop('origin_key', 'o')
        origin = tuple(f[origin_key][()])

    spacing = kwargs.pop('spacing', None)
    if spacing is None:
        spacing_key = kwargs.pop('spacing_key', 'd')
        spacing = tuple(f[spacing_key][()])

    nbpml = kwargs.pop('nbpml', 20)
    datakey = kwargs.pop('datakey', None)
    if datakey is None:
        raise ValueError("datakey must be known - what is the name of the data in the file?")

    space_order=kwargs.pop('space_order', None)
    dtype = kwargs.pop('dtype', None)
    data_m = f[datakey][()]
    data_vp = np.sqrt(1/data_m).astype(dtype)
    data_vp = np.transpose(data_vp, (1, 2, 0))
    shape = data_vp.shape

    f.close()

    return Model(space_order=space_order, vp=data_vp, origin=origin, shape=shape,
                 dtype=dtype, spacing=spacing, nbpml=nbpml)

def get_true_model():
    filename = 'overthrust_3D_true_model.h5'

    model_file = Path(filename)
    if not model_file.is_file():
        download_file_from_bucket(filename)

    return from_hdf5(filename, nbpml=param['nbpml'], space_order=4,
                     datakey='m', dtype=np.float32)

def dump_shot_data(shot_id, src, rec):
    ''' Dump shot data to disk.
    '''
    filename = 'shot_%d.p'%shot_id
    pickle.dump({'src':src, 'rec':rec}, open(filename, "wb"))

    upload_file_to_bucket(filename)

def generate_shotdata_i(param):
    from devito import clear_cache

    os.environ['DEVITO_OPENMP'] = "1"

    # Need to clear the workers cache.
    clear_cache()

    true_model = get_true_model()
    shot_id = param['shot_id']

    param['shape'] = true_model.vp.shape
    param['spacing'] = true_model.spacing
    param['origin'] = true_model.origin

    i = shot_id%97
    j = int(shot_id/97)

    # Time step from model grid spacing
    dt = true_model.critical_dt

    # Set up source data and geometry.
    time_range = TimeAxis(start=param['t0'], stop=param['tn'], step=dt)
    src = RickerSource(name='src', grid=true_model.grid, f0=param['f0'],
                       time_range=time_range)

    src.coordinates.data[0, :] = [400+i*4*true_model.spacing[0], 400+j*4*true_model.spacing[1], 50]

    # Number of receiver locations per shot.
    nreceivers = 97**2

    # Set up receiver data and geometry.
    rec = Receiver(name='rec', grid=true_model.grid, time_range=time_range,
                   npoint=nreceivers)

    for n in range(97):
        for m in range(97):
            rec.coordinates.data[:, 0] = 400+n*4*true_model.spacing[0]
            rec.coordinates.data[:, 1] = 400+m*4*true_model.spacing[1]
            rec.coordinates.data[:, 2] = 50

    # Set up solver.
    solver = AcousticWaveSolver(true_model, src, rec, space_order=4)

    # Generate synthetic receiver data from true model.
    true_d, _, _ = solver.forward(src=src, m=true_model.m)

    dump_shot_data(shot_id, src, true_d)

def generate_shotdata(param):
    # Define work list
    work = [dict(param) for i in range(param['nshots'])]
    for i in  range(param['nshots']):
        work[i]['shot_id'] = i

    # Map worklist to cluster
    for i in range(0, param['nshots'], len(client.ncores())):
            futures = client.map(generate_shotdata_i, work[i:i+len(client.ncores())])

            # Wait for all futures
            wait(futures)


if __name__ == "__main__":
    client = Client(kwargs={'n_workers':1, 'threads_per_worker':1,
                            'death_timeout':600, 'memory_limit':64e9})

    generate_shotdata(param)

    client.close()

