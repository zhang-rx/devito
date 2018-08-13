from devito.dle import AdvancedRewriter
from devito.dle.backends import dle_pass
from devito.ir.iet import FindNodes, Expression, Iteration, Transformer
from devito.ir.iet.visitors import Visitor
from sympy import Symbol
from devito import Eq
from devito.ir.equations.equation import ClusterizedEq
from devito.cgen_utils import ccode

class Assignment(object):
    def __init__(self, var, space=None):
        self.var = var
        self.space = space
        self.ranges = {'t': 't1', 'x':'x', 'y':'y'}
        
    def __str__(self):
        if type(self.var) is str:
            return self.var
        else:
            s = self.var.name
            for x in self.var.indices:
                r = self.ranges[str(x)]
                if type(r) is tuple:
                    r = str(r[0]) + "..." + str(r[1])
                s += "[%s]"%r
            return s

def update_writes(writes, iteration):
    for w in writes:
        assert(type(w) is Assignment)
        if hasattr(w.var, 'is_Tensor'):
            w.ranges[iteration.index] = iteration.limits
    writes += [Assignment(iteration.index)]
    return writes

    
def get_written_variables(node):
    assert(node.is_Iteration)
    writes = []

    for n in node.nodes:
        if n.is_ExpressionBundle:
            writes.extend([Assignment(k, s) for (k, v), s in n.traffic.items() if v=='w'])
        elif n.is_Iteration:
            writes.extend(get_written_variables(n))
    writes = update_writes(writes, node)
    return writes

class LoopAnnotation(object):
    annotation_types = ['invariant', 'assigns']
    def __init__(self, annotation_type, content):
        assert(annotation_type in self.annotation_types)
        self.annotation_type = annotation_type
        self.content = content

    def __str__(self):
        return "@ loop %s %s;" % (self.annotation_type, self.content)


class LoopDefinitionInvariant(object):
    def __init__(self, node, children=None):
        self.node = node
        self.children = children
        self.eq = None
        if type(node) is ClusterizedEq or type(node) is Eq:            
            t = node.lhs.args[1] - 1
            t1 = Symbol('t1')
            t0 = Symbol('t0')
            self.eq = Eq(node.lhs.subs(t+1, t1), node.rhs.subs(t, t0))

    def subs(self, old, new):
        if self.eq is not None:
            return LoopDefinitionInvariant(self.eq.subs(old, new))
        else:
            return self.children.subs(old, new)

    def __str__(self):
        if self.eq is not None:
            return ccode(self.eq)
        else:
            return "\\forall int %s1; %s <= %s1 <= %s ==> (%s)" % (self.node.index, self.node.limits[0], self.node.index, self.node.index, str(self.children).replace(self.node.index, '%s1'%str(self.node.index)))


class LoopInvariantVisitor(Visitor):
    def visit_Iteration(self,o):
        return LoopDefinitionInvariant(o, children=self.visit(o.nodes[0]))
        #return "forall int %s1; %s <= %s1 <= %s + 1 ==> (%s)" % (o.index, o.limits[0], o.index, o.limits[1], )
        

    def visit_ExpressionBundle(self, o):
        return LoopDefinitionInvariant(o.orig)


class LoopAnnotationRewriter(AdvancedRewriter):
    def _pipeline(self, state):
        self._extract_invariants(state)

    @dle_pass
    def _extract_invariants(self, nodes, state):
        iterations = FindNodes(Iteration).visit(nodes)
        mapper = {}
        for i in iterations:
            if i.is_Sequential:
                continue
            original_a = i.annotations or []
            writes = [str(x) for x in get_written_variables(i)]
            inv = LoopInvariantVisitor().visit(i)
            a = []
            a.append(LoopAnnotation('invariant', "%s <= %s <= %s + 1" % (i.limits[0], i.index, i.limits[1])))
            a.append(LoopAnnotation('invariant', "(%s - %s)%%%s == 0" % (i.index, i.limits[0], i.limits[2])))
            a.append(LoopAnnotation('invariant', str(inv)))
            a.append(LoopAnnotation('assigns', ", ".join(writes)))

            
            mapper[i] = i._rebuild(annotations=([str(x) for x in a] + original_a))

        processed = Transformer(mapper).visit(nodes)
        return processed, {}


