from devito.operator import OperatorRunnable
from sympy import Eq


class Operator(OperatorRunnable):

    """
    A special :class:`OperatorCore` to JIT-compile and run operators through YASK.
    """

    _default_headers = OperatorRunnable._default_headers

    def __init__(self, expressions, **kwargs):
        super(Operator, self).__init__(expressions, **kwargs)


    def _specialize_exprs(self, expressions):

        expressions = super(Operator, self)._specialize_exprs(expressions)

        for e in expressions:
            e.original = Eq(e.lhs, e.rhs)
        return expressions

    
    #def _specialize_iet(self, iet, **kwargs):
         
        
