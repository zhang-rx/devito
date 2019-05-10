#!/usr/bin/env bash

find /home/devito -type f -name '*.pyc' -delete

export PATH=/home/devito/.conda/envs/devito/bin:/usr/local/miniconda/condabin:$PATH
export PYTHONPATH=$PYTHONPATH:/home/devito/.conda/envs/devito

exec "$@"
