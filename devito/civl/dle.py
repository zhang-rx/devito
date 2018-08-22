from devito.dle import AdvancedRewriter
from devito.dle.backends import dle_pass
from devito.ir.iet import FindNodes, Expression, Iteration, Transformer, ExpressionBundle
from devito.ir.iet.visitors import Visitor
from sympy import Symbol
from devito import Eq
from devito.ir.equations.equation import ClusterizedEq
from devito.cgen_utils import ccode
from devito.tools.utils import flatten
import copy

class Assignment(object):
    def __init__(self, var, space=None):
        self.var = var
        self.space = space
        self.ranges = {'t': 't1', 'x':'x', 'y':'y', 'x0_block': 'x', 'y0_block': 'y'}
        
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
            if hasattr(iteration.dim, 'parent'):
                w.ranges[str(iteration.dim.parent)] = iteration.limits
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
        else:
            self.limits = {'lower': self.node.limits[0], 'upper': self.node.index, 'index': Symbol("%s1"%self.node.index)}

    def subs(self, old, new):
        if self.eq is not None:
            return LoopDefinitionInvariant(self.eq.subs(old, new))
        else:
            cp = copy.copy(self)
            cp.children = cp.children.subs(old, new)
            return cp

    def msubs(self, subs):
        ret = self
        for k, v in subs.items():
            ret = ret.subs(k, v)
        return ret

    def __str__(self):
        if self.eq is not None:
            return ccode(self.eq)
        else:
            if type(self.children.node) is Iteration:
                self.children.limits['upper'] = self.children.node.limits[1]
            subs = {self.node.index: self.limits['index']}
            if self.node.dim.is_Derived:
                subs[self.node.dim.parent] = self.limits['index']
            child = self.children.msubs(subs)
                
            return "\\forall int %s; %s <= %s < %s ==> (%s)" % (ccode(self.limits['index']), ccode(self.limits['lower']), ccode(self.limits['index']), ccode(self.limits['upper']), str(child))


class DefinitionInvariantVisitor(Visitor):
    def visit_Iteration(self,o):
        return LoopDefinitionInvariant(o, children=self.visit(o.nodes[0]))

    def visit_ExpressionBundle(self, o):
        return LoopDefinitionInvariant(o.orig)

    def visit_List(self, o):
        return self.visit(o.body[0])

    def visit_Call(self, o):
        return self.visit(o.called).msubs(dict(zip(o.called.parameters, o.params)))

    def visit_Callable(self, o):
        expressions = FindNodes(ExpressionBundle).visit(o.body)
        return LoopDefinitionInvariant(expressions[0].orig)


class AssignInvariantVisitor(Visitor):
    def visit_Iteration(self, o):
        return self.visit(o.nodes)

    def visit_ExpressionBundle(self, o):
        return [Assignment(k, s) for (k, v), s in o.traffic.items() if v=='w']

    def visit_Call(self, o):
        return self.visit(o.called)

    def visit_Callable(self, o):
        return self.visit(o.body)

    def visit_tuple(self, o):
        return flatten([self.visit(x) for x in o])

class LoopAnnotationRewriter(AdvancedRewriter):
    def _pipeline(self, state):
        super(LoopAnnotationRewriter, self)._pipeline(state)
        self._extract_invariants(state)

    @dle_pass
    def _extract_invariants(self, nodes, state):
        iterations = FindNodes(Iteration).visit(nodes)
        mapper = {}
        for i in iterations:
            if i.is_Sequential:
                continue
            original_a = i.annotations or []
            writes = AssignInvariantVisitor().visit(i)
            writes = update_writes(writes, i)
            writes += [Assignment(i.index)]
            inv = DefinitionInvariantVisitor().visit(i)
            a = []
            a.append(LoopAnnotation('invariant', "%s <= %s <= %s + 1" % (ccode(i.limits[0]), ccode(i.index), ccode(i.limits[1]))))
            a.append(LoopAnnotation('invariant', "(%s - %s)%%%s == 0" % (ccode(i.index), ccode(i.limits[0]), ccode(i.limits[2]))))
            a.append(LoopAnnotation('invariant', str(inv)))
            a.append(LoopAnnotation('assigns', ", ".join([str(x) for x in writes])))

            
            mapper[i] = i._rebuild(annotations=([str(x) for x in a] + original_a))

        processed = Transformer(mapper).visit(nodes)
        return processed, {}


