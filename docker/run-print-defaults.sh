#!/usr/bin/env bash

PYTHONPATH=/home/devito/.conda/envs/devito /home/devito/.conda/envs/devito/bin/python -c "
from devito import print_defaults;
print_defaults();
"
