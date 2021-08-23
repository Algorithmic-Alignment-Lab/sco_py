import numdifftools as nd
import numpy as np
from scipy.linalg import eigvalsh

DEFAULT_TOL = 1e-4

"""
Utility classes to represent expresions. Each expression defines an eval, grad,
hess, and convexify method. Variables and values are assumed to be 2D numpy
arrays.
"""

N_DIGS = 6  # 10


class Expr(object):

    """
    by default, expressions are defined by black box functions
    """

    def __init__(self, f, grad=None, hess=None, **kwargs):
        self.f = f
        self._grad = grad
        self._hess = hess

        self._eval_cache = {}
        self._grad_cache = {}
        self._convexify_cache = {}

    def _get_key(self, x):
        return tuple(x.round(N_DIGS).flatten())

    def eval(self, x):
        key = self._get_key(x)

        if key in self._eval_cache:
            return self._eval_cache[key]
        val = self.f(x)
        self._eval_cache[key] = val.copy()
        return val

    def _get_flat_f(self, x):
        """
        Utility function which reshapes and flattens for compatibility with
        numdifftools' Jacobian and Hessian function.
        """
        if len(x.shape) == 2:
            rows, cols = x.shape
            assert cols == 1

            def flat_f(x):
                return self.f(x.reshape((rows, cols))).flatten()

            return flat_f
        elif len(x.shape) == 1:
            return self.f
        else:
            raise Exception("Input shape not supported")

    def _num_grad(self, x):
        """
        Returns a numerically computed gradient.
        Flattening is necessary for compatibility with numdifftools' Jacobian
        function.
        """
        grad_fn = nd.Jacobian(self._get_flat_f(x))

        return grad_fn(x)

    def _debug_grad(self, g1, g2, atol=DEFAULT_TOL):
        for i, g_row in enumerate(g1):
            for j, g in enumerate(g_row):
                if not np.allclose(g, g2[i, j], atol=atol):
                    print("{}, {}".format(i, j))
                    print(g, g2[i, j])

    def grad(self, x, num_check=False, atol=DEFAULT_TOL):
        """
        Returns the gradient. Can numerically check the gradient.
        """
        key = self._get_key(x)
        if key in self._grad_cache:
            return self._grad_cache[key].copy()
        assert not num_check or self._grad is not None
        if self._grad is None:
            return self._num_grad(x)
        gradient = self._grad(x)
        if num_check:
            num_grad = self._num_grad(x)
            if not np.allclose(num_grad, gradient, atol=atol):
                self._debug_grad(gradient, num_grad, atol=atol)
                raise Exception(
                    "Numerical and analytical gradients aren't close. \
                    \nnum_grad: {0}\nana_grad: {1}\n".format(
                        num_grad, gradient
                    )
                )
        self._grad_cache[key] = gradient.copy()
        return gradient

    def _num_hess(self, x):
        """
        Returns a numerically computed hessian.
        Flattening is necessary for compatibility with numdifftools' Hessian
        function.
        """
        hess_fn = nd.Hessian(self._get_flat_f(x))
        return hess_fn(x.flatten())

    def hess(self, x, num_check=False, atol=DEFAULT_TOL):
        """
        Returns the hessian. Can numerically check the hessian.
        """
        assert not num_check or self._hess is not None
        if self._hess is None:
            return self._num_hess(x)
        hessian = self._hess(x)
        if num_check:
            num_hess = self._num_hess(x)
            if not np.allclose(num_hess, hessian, atol=atol):
                raise Exception(
                    "Numerical and analytical hessians aren't close. \
                    \nnum_hess: {0}\nana_hess: {1}\n".format(
                        num_hess, hessian
                    )
                )
        return hessian

    def convexify(self, x, degree=1):
        """
        Returns an Expression object that represents the convex approximation of
        self at x where degree 1 is an affine approximation and degree 2 is a
        quadratic approximation. If the hessian has negative eigenvalues, the
        hessian is adjusted so that it is positive semi-definite.
        """

        res = None
        if degree == 1:
            A = self.grad(x)
            b = -A.dot(x) + self.eval(x)
            res = AffExpr(A, b)
        elif degree == 2:
            hess = self.hess(x)
            eig_vals = eigvalsh(hess)
            min_eig_val = min(eig_vals)
            if min_eig_val < 0:
                hess = hess - np.eye(hess.shape[0]) * min_eig_val
            grad = self.grad(x)
            Q = hess
            A = grad - np.transpose(x).dot(hess)
            b = 0.5 * np.transpose(x).dot(hess).dot(x) - grad.dot(x) + self.eval(x)
            res = QuadExpr(Q, A, b)
        else:
            raise NotImplementedError
        return res


class AffExpr(Expr):
    """
    Affine Expression
    """

    def __init__(self, A, b):
        """
        expr is Ax + b
        """
        assert b.shape[0] == A.shape[0]

        self.A = A
        self.b = b
        self.x_shape = (A.shape[1], 1)

    def eval(self, x):
        return self.A.dot(x) + self.b

    def grad(self, x):
        return self.A.T

    def hess(self, x):
        return np.zeros((self.x_shape[0], self.x_shape[0]))


class QuadExpr(Expr):
    """
    Quadratic Expression
    """

    def __init__(self, Q, A, b):
        """
        expr is 0.5*x'Qx + Ax + b
        """
        assert A.shape[0] == 1, "Can only define scalar quadrative expressions"

        # ensure the correct shapes for all the arguments
        assert Q.shape[0] == Q.shape[1]
        assert Q.shape[0] == A.shape[1]
        assert b.shape[0] == 1

        self.Q = Q
        self.A = A
        self.b = b
        self.x_shape = (A.shape[1], 1)

    def eval(self, x):
        return 0.5 * x.T.dot(self.Q.dot(x)) + self.A.dot(x) + self.b

    def grad(self, x):
        assert x.shape == self.x_shape
        return 0.5 * (self.Q.dot(x) + self.Q.T.dot(x)) + self.A.T

    def hess(self, x):
        return self.Q.copy()


class AbsExpr(Expr):
    """
    Absolute value expression
    """

    def __init__(self, expr):
        self.expr = expr

    def eval(self, x):
        return np.absolute(self.expr.eval(x))

    def grad(self, x):
        """
        Since the absolute value expression is not smooth, a subgradient is
        returned instead of the gradient.
        """
        raise NotImplementedError

    def hess(self, x):
        raise NotImplementedError


class HingeExpr(Expr):
    """
    Hinge expression
    """

    def __init__(self, expr):
        self.expr = expr

    def eval(self, x):
        v = self.expr.eval(x)
        zeros = np.zeros(v.shape)
        return np.maximum(v, zeros)

    def grad(self, x):
        """
        Since the hinge expression is not smooth, a subgradient is returned
        instead of the gradient.
        """
        raise NotImplementedError

    def hess(self, x):
        raise NotImplementedError


class CompExpr(Expr):
    """
    Comparison Expression
    """

    def __init__(self, expr, val):
        """
        expr: Expr object, the expression that is being compared to val
        val: numpy array, the value that the expression is being compared to
        """
        self.expr = expr
        self.val = val.copy()
        self._convexify_cache = {}

    def eval(self, x, tol=DEFAULT_TOL):
        """
        Returns True if the comparison holds true within some tolerace and 0
        otherwise.
        """
        raise NotImplementedError

    def grad(self, x):
        raise Exception(
            "The gradient is not well defined for comparison \
            expressions"
        )

    def hess(self, x):
        raise Exception(
            "The hessian is not well defined for comparison \
            expressions"
        )

    def convexify(self, x, degree=1):
        raise NotImplementedError


class EqExpr(CompExpr):
    """
    Equality Expression
    """

    def eval(self, x, tol=DEFAULT_TOL, negated=False):
        """
        Tests whether the expression at x is equal to self.val with tolerance
        tol.
        """
        assert tol >= 0.0
        if negated:
            return not np.allclose(self.expr.eval(x), self.val, atol=tol)
        return np.allclose(self.expr.eval(x), self.val, atol=tol)

    def convexify(self, x, degree=1):
        """
        Returns an AbsExpr that is the l1 penalty expression, a measure of
        constraint violation.

        The constraint h(x) = 0 becomes |h(x)|
        """
        assert degree == 1

        key = self._get_key(x)
        if key in self._convexify_cache:
            return self._convexify_cache[key]

        aff_expr = self.expr.convexify(x, degree=1)
        aff_expr.b = aff_expr.b - self.val
        res = AbsExpr(aff_expr)

        self._convexify_cache[key] = res
        return res


class LEqExpr(CompExpr):
    """
    Less than or equal to expression
    """

    def eval(self, x, tol=DEFAULT_TOL, negated=False):
        """
        Tests whether the expression at x is less than or equal to self.val with
        tolerance tol.
        """
        assert tol >= 0.0
        expr_val = self.expr.eval(x)
        if negated:
            # need the tolerance to go the other way if its negated
            return not np.all(expr_val <= self.val - tol * np.ones(expr_val.shape))
        else:
            return np.all(expr_val <= self.val + tol * np.ones(expr_val.shape))

    def convexify(self, x, degree=1):
        """
        Returns a HingeExpr that is the hinge penalty expression, a measure of
        constraint violation.

        The constraint g(x) <= 0 becomes |g(x)|+ where |g(x)|+ = max(g(x), 0)
        """
        assert degree == 1

        key = self._get_key(x)
        if key in self._convexify_cache:
            return self._convexify_cache[key]

        aff_expr = self.expr.convexify(x, degree=1)
        aff_expr.b = aff_expr.b - self.val
        res = HingeExpr(aff_expr)

        self._convexify_cache[key] = res
        return res


class LExpr(CompExpr):
    """
    Less than expression
    """

    def eval(self, x, tol=DEFAULT_TOL, negated=False):
        """
        Tests whether the expression at x is less than or equal to self.val with
        tolerance tol.
        """
        assert tol >= 0.0
        expr_val = self.expr.eval(x)
        if negated:
            # need the tolerance to go the other way if its negated
            return not np.all(expr_val < self.val - tol * np.ones(expr_val.shape))
        else:
            return np.all(expr_val < self.val + tol * np.ones(expr_val.shape))

    def convexify(self, x, degree=1):
        """
        Returns a HingeExpr that is the hinge penalty expression, a measure of
        constraint violation.

        The constraint g(x) <= 0 becomes |g(x)|+ where |g(x)|+ = max(g(x), 0)
        """
        assert degree == 1

        key = self._get_key(x)
        if key in self._convexify_cache:
            return self._convexify_cache[key]

        aff_expr = self.expr.convexify(x, degree=1)
        aff_expr.b = aff_expr.b - self.val
        res = HingeExpr(aff_expr)

        self._convexify_cache[key] = res
        return res


class BoundExpr(object):
    """
    Bound expression

    Bound expression is composed of an Expr and a Variable. Please note that the
    variable ordering matters
    """

    def __init__(self, expr, var):
        self.expr = expr
        self.var = var

    def eval(self):
        """
        Returns the current value of the bound expression
        """
        return self.expr.eval(self.var.get_value())

    def convexify(self, degree=1):
        """
        Returns a convexified BoundExpr at the variable's current value.
        """
        assert self.var.get_value() is not None
        cvx_expr = self.expr.convexify(self.var.get_value(), degree)
        return BoundExpr(cvx_expr, self.var)


class TFExpr(Expr):

    """
    TODO

    wrapper around exprs defined by a tensorflow graph. Leverages
    automated differentition.
    """

    def __init__(self, f, grad=None, hess=None, sess=None):
        self.sess = sess
        return super(TFExpr, self).__init__(f, grad, hess)
