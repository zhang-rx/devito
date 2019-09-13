import os

import click

from benchmark import option_simulation
import devito


@click.group()
def menu():
    pass


@menu.command(name='generate')
@option_simulation
@click.option('-nn', multiple=True, default=[1], help='Number of nodes')
@click.option('-nt', default=1, help='Number of OpenMP threads *per MPI process*')
@click.option('--mpi', multiple=True, default=['basic'], help='Devito MPI mode(s)')
@click.option('--arch', default='unknown', help='Test-bed architecture')
@click.option('-r', '--resultsdir', default='results', help='Results directory')
@click.option('--export', multiple=True, default=[], help='Env vars to be exported')
def generate(**kwargs):
    join = lambda l: ' '.join('%d' % i for i in l)
    args = dict(kwargs)
    args['shape'] = join(args['shape'])
    args['space_order'] = join(args['space_order'])

    args['home'] = os.path.dirname(os.path.dirname(devito.__file__))

    args['export'] = '\n'.join('export %s' % i for i in args['export'])

    template_header = """\
#!/bin/bash

#rj queue=idle nodes=%(nn)s priority=900 logdir=logs mem=110G io=0 features=xeon

lscpu

module load intel-composer/2019.0.117
module load intel-rt/2019.0.117
module load openmpi/3.0.0-mt

module load miniconda  # otherwise "activate" won't work

cd %(home)s

source activate devito

export DEVITO_HOME=%(home)s
export DEVITO_ARCH=intel
export DEVITO_OPENMP=1
export DEVITO_LOGGING=DEBUG

export OMP_NUM_THREADS=%(nt)s
export OMP_PLACES=cores
export OMP_PROC_BIND=close

%(export)s

cd benchmarks/user
"""  # noqa
    template_cmd = """\
DEVITO_MPI=%(mpi)s mpirun -np %(np)s --bind-to socket --report-bindings python benchmark.py bench -P %(problem)s -bm O2 -d %(shape)s -so %(space_order)s --tn %(tn)s -x 1 --arch %(arch)s -r %(resultsdir)s\
"""  # noqa

    # Generate one PBS file for each `np` value
    for nn in kwargs['nn']:
        args['nn'] = nn
        args['np'] = str(int(nn)*2)

        cmds = []
        for i in kwargs['mpi']:
            args['mpi'] = i
            cmds.append(template_cmd % args)
        cmds = ' \n'.join(cmds)

        body = ' \n'.join([template_header % args, cmds])

        with open('pbs_nn%d.gen.sh' % int(nn), 'w') as f:
            f.write(body)


@menu.command(name='cleanup')
def cleanup():
    for f in os.listdir():
        if f.endswith('.gen.sh'):
            os.remove(f)


if __name__ == "__main__":
    menu()
