from functools import partial

from devito.core.operator import OperatorCore
from devito.exceptions import InvalidOperator
from devito.ir.clusters import Toposort
from devito.passes.clusters import (Blocking, Interchange, Lift, cire, cse,
                                    eliminate_arrays, extract_increments, factorize,
                                    freeze, fuse, optimize_pows, scalarize)
from devito.passes.iet import (DataManager, Ompizer, avoid_denormals, mpiize,
                               optimize_halospots, loop_wrapping, hoist_prodders,
                               relax_incr_dimensions)
from devito.tools import as_tuple, generator, timed_pass

__all__ = ['CPU64NoopOperator', 'CPU64Operator', 'Intel64Operator', 'PowerOperator',
           'ArmOperator', 'CustomOperator']


class CPU64NoopOperator(OperatorCore):

    BLOCK_LEVELS = 1
    """
    Loop blocking depth. So, 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']

        # Distributed-memory parallelism
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Shared-memory parallelism
        if options['openmp']:
            ompizer = Ompizer()
            ompizer.make_parallel(graph)

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


class CPU64Operator(CPU64NoopOperator):

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        """
        Optimize Clusters for better runtime performance.
        """
        options = kwargs['options']
        platform = kwargs['platform']

        # To create temporaries
        counter = generator()
        template = lambda: "r%d" % counter()

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = Toposort().process(clusters)
        clusters = fuse(clusters)

        # Hoist and optimize Dimension-invariant sub-expressions
        clusters = cire(clusters, template, platform, 'invariants')
        clusters = Lift().process(clusters)

        # Blocking to improve data locality
        inner = options['blockinner']
        levels = options['blocklevels'] or cls.BLOCK_LEVELS
        clusters = Blocking(inner, levels).process(clusters)

        # Interchange to maximize collapsing and minimize synchronization points
        clusters = Interchange().process(clusters)

        # Reduce flops
        clusters = extract_increments(clusters, template)
        clusters = cire(clusters, template, platform, 'sops')
        #TODO
        clusters = cire(clusters, template, platform, 'sops')
        clusters = factorize(clusters)
        clusters = cse(clusters, template)
        clusters = optimize_pows(clusters)
        clusters = freeze(clusters)

        # The previous passes may have created fusion opportunities, which in
        # turn may enable further optimizations
        clusters = fuse(clusters)
        clusters = eliminate_arrays(clusters, template)
        clusters = scalarize(clusters, template)

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']

        # Flush denormal numbers
        avoid_denormals(graph)

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Lower IncrDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph)

        # Shared-memory and SIMD-level parallelism
        ompizer = Ompizer()
        ompizer.make_simd(graph, simd_reg_size=platform.simd_reg_size)
        if options['openmp']:
            ompizer.make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol lowering and definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph


Intel64Operator = CPU64Operator
PowerOperator = CPU64Operator
ArmOperator = CPU64Operator


class CustomOperator(CPU64Operator):

    _known_passes = ('blocking', 'denormals', 'optcomms', 'wrapping', 'openmp',
                     'mpi', 'simd', 'prodders', 'topofuse', 'toposort', 'fuse')

    @classmethod
    def _make_clusters_passes_mapper(cls, **kwargs):
        options = kwargs['options']

        inner = options['blockinner']
        levels = options['blocklevels'] or cls.BLOCK_LEVELS

        return {
            'toposort': Toposort().process,
            'fuse': fuse,
            'blocking': Blocking(inner, levels).process,
            # Pre-baked composite passes
            'topofuse':\
                lambda i: fuse(Toposort().process(i))
        }


    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']

        ompizer = Ompizer()

        return {
            'denormals': avoid_denormals,
            'optcomms': optimize_halospots,
            'wrapping': loop_wrapping,
            'blocking': relax_incr_dimensions,
            'openmp': ompizer.make_parallel,
            'mpi': partial(mpiize, mode=options['mpi']),
            'simd': partial(ompizer.make_simd, simd_reg_size=platform.simd_reg_size),
            'prodders': hoist_prodders
        }

    @classmethod
    def _compile(cls, expressions, **kwargs):
        #TODO: before merging, update this to _build

        # Sanity check
        passes = as_tuple(kwargs['mode'])
        if any(i not in cls._known_passes for i in passes):
            raise InvalidOperator("Unknown passes `%s`" % str(passes))

        return super(CustomOperator, cls)._compile(expressions, **kwargs)

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_clusters_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                clusters = passes_mapper[i](clusters)
            except KeyError:
                pass

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_iet_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                passes_mapper[i](graph)
            except KeyError:
                pass

        # Force-call `mpi` if requested via global option
        if 'mpi' not in passes and options['mpi']:
            passes_mapper['mpi'](graph)

        # Force-call `openmp` if requested via global option
        if 'openmp' not in passes and options['openmp']:
            passes_mapper['openmp'](graph)

        # Symbol definitions
        data_manager = DataManager()
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        return graph
