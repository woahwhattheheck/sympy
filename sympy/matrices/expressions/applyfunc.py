from __future__ import annotations
from sympy.core.function import (Function, FunctionClass, Lambda)
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.matrices.matrixbase import MatrixBase


class ElementwiseApplyFunction(MatrixExpr):
    r"""
    Apply function to a matrix elementwise without evaluating.

    Examples
    ========

    It can be created by calling ``.applyfunc(<function>)`` on a matrix
    expression:

    >>> from sympy import MatrixSymbol
    >>> from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
    >>> from sympy import exp
    >>> X = MatrixSymbol("X", 3, 3)
    >>> X.applyfunc(exp)
    Lambda(_d, exp(_d)).(X)

    Otherwise using the class constructor:

    >>> from sympy import eye
    >>> expr = ElementwiseApplyFunction(exp, eye(3))
    >>> expr
    Lambda(_d, exp(_d)).(Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]))
    >>> expr.doit()
    Matrix([
    [E, 1, 1],
    [1, E, 1],
    [1, 1, E]])

    Notice the difference with the real mathematical functions:

    >>> exp(eye(3))
    Matrix([
    [E, 0, 0],
    [0, E, 0],
    [0, 0, E]])
    """

    def __new__(cls, function, expr):
        # Array differentiation imports this class during package initialization.
        from sympy.tensor.array.expressions.utils import _avoid_bound_sum_capture

        expr = _sympify(expr)
        if not expr.is_Matrix:
            raise ValueError("{} must be a matrix instance.".format(expr))

        function = _avoid_bound_sum_capture(function, expr)
        if expr.shape == (1, 1):
            # Some scalar maps also have a matrix representation, e.g. 2*d.
            # This is only an optimization: a valid scalar map such as
            # d**p*log(d) need not accept a matrix as its argument.
            try:
                ret = function(expr)
            except (TypeError, NotImplementedError):
                pass
            else:
                if isinstance(ret, MatrixExpr):
                    return ret

        if not isinstance(function, (FunctionClass, Lambda)):
            d = Dummy('d')
            function = Lambda(d, function(d))

        function = sympify(function)
        if not isinstance(function, (FunctionClass, Lambda)):
            raise ValueError(
                "{} should be compatible with SymPy function classes."
                .format(function))

        if 1 not in function.nargs:
            raise ValueError(
                '{} should be able to accept 1 arguments.'.format(function))

        if not isinstance(function, Lambda):
            d = Dummy('d')
            function = Lambda(d, function(d))

        function = _avoid_bound_sum_capture(function, expr)
        obj = MatrixExpr.__new__(cls, function, expr)
        return obj

    @property
    def function(self):
        return self.args[0]

    @property
    def expr(self):
        return self.args[1]

    @property
    def shape(self):
        return self.expr.shape

    def doit(self, **hints):
        deep = hints.get("deep", True)
        expr = self.expr
        if deep:
            expr = expr.doit(**hints)
        function = self.function
        if isinstance(function, Lambda) and function.is_identity:
            # This is a Lambda containing the identity function.
            return expr
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(self.function)
        elif isinstance(expr, ElementwiseApplyFunction):
            return ElementwiseApplyFunction(
                lambda x: self.function(expr.function(x)),
                expr.expr
            ).doit(**hints)
        else:
            return self

    def _entry(self, i, j, **kwargs):
        return self.function(self.expr._entry(i, j, **kwargs))

    def _get_function_fdiff(self):
        d = Dummy("d")
        function = self.function(d)
        fdiff = function.diff(d)
        if isinstance(fdiff, Function) and fdiff.args == (d,):
            # Keep the bare function class only when the derivative is the
            # function applied to the same variable; otherwise the argument
            # (e.g. the shift in exp(d + 1)) would be lost.
            fdiff = type(fdiff)
        else:
            fdiff = Lambda(d, fdiff)
        return fdiff

    def _eval_derivative(self, x):
        from sympy.matrices.expressions.hadamard import hadamard_product
        dexpr = self.expr.diff(x)
        fdiff = self._get_function_fdiff()
        result = hadamard_product(
            dexpr,
            ElementwiseApplyFunction(fdiff, self.expr)
        )
        d = Dummy("d")
        partial = self.function(d).diff(x)
        if partial != 0:
            result += ElementwiseApplyFunction(Lambda(d, partial), self.expr)
        return result

    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import Transpose
        return self.func(self.function, Transpose(self.expr).doit())
