from devito.dle import AdvancedRewriter
from devito.dle.backends import dle_pass

class LoopAnnotationRewriter(AdvancedRewriter):
    def _pipeline(self, state):
        self._extract_invariants(state)

    @dle_pass
    def _extract_invariants(self, nodes, state):
        iterations = FindNodes(Iteration).visit(nodes)
        print("hello")
