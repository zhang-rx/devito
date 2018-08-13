from devito.dle import (BasicRewriter, AdvancedRewriter, AdvancedRewriterSafeMath,
                        SpeculativeRewriter, init_dle)
from devito.parameters import Parameters, add_sub_configuration
from .dle import LoopAnnotationRewriter


core_configuration = Parameters('civl')
env_vars_mapper = {}
add_sub_configuration(core_configuration, env_vars_mapper)

# Initialize the DLE
modes = {'basic': BasicRewriter,
         'advanced': LoopAnnotationRewriter,
         'advanced-safemath': AdvancedRewriterSafeMath,
         'speculative': SpeculativeRewriter}
init_dle(modes)

# The following used by backends.backendSelector
from devito.function import *  # noqa
from devito.grid import Grid  # noqa
from .operator import Operator
from devito.types import CacheManager  # noqa
