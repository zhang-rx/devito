
def external_initializer(x):
    return None


def func_args(equation):
    """
    Retrieve all Functions from a given Equation.
    """
    if equation.is_Function:
        return equation
    elif equation.is_Equality:
        return [func_args(equation.lhs),
                func_args(equation.rhs)]
    elif equation.is_Add:
        return [func_args(i) for i in equation.args if not(i.is_Number)]
    else:
        error_string = "Unimplemented operation \n\t%s:  "\
                       "\nfor case deteceted in equation: \n\t< %s >" \
                       % (equation.func, equation)
        raise(ValueError(error_string))
