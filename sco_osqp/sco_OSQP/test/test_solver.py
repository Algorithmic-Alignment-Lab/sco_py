import unittest

import numpy as np

from sco_OSQP.expr import AffExpr, BoundExpr, EqExpr, Expr, LEqExpr, QuadExpr
from sco_OSQP.osqp_utils import OSQPVar
from sco_OSQP.prob import Prob
from sco_OSQP.solver import Solver
from sco_OSQP.variable import Variable

solv = Solver()
"""
values taken from Pieter Abbeel's CS287 hw3 q2 penalty_sqp.m file
"""
solv.improve_ratio_threshold = 0.25
solv.min_trust_region_size = 1e-5
solv.min_approx_improve = 1e-8
solv.max_iter = 50
solv.trust_shrink_ratio = 0.1
solv.trust_expand_ratio = 1.5
solv.cnt_tolerance = 1e-4
solv.max_merit_coeff_increases = 5
solv.merit_coeff_increase_ratio = 10
solv.initial_trust_region_size = 1
solv.initial_penalty_coeff = 1.0

zerofunc = lambda x: np.array([[0.0]])
neginffunc = lambda x: np.array([[-1e5]])
N = 2


def helper_test_prob(
    ut,
    x0,
    x_true,
    f=zerofunc,
    g=neginffunc,
    h=zerofunc,
    Q=np.zeros((N, N)),
    q=np.zeros((1, N)),
    A_ineq=np.zeros((1, N)),
    b_ineq=np.zeros((1, 1)),
    A_eq=np.zeros((1, 1)),
    b_eq=np.zeros((1, 1)),
):

    if not np.allclose(A_eq, np.zeros((1, 1))) or not np.allclose(
        b_eq, np.zeros((1, 1))
    ):
        raise NotImplementedError

    prob = Prob()

    osqp_var1 = OSQPVar("x1")
    osqp_var2 = OSQPVar("x2")
    prob.add_osqp_var(osqp_var1)
    prob.add_osqp_var(osqp_var2)
    osqp_vars = np.array([[osqp_var1], [osqp_var2]])
    var = Variable(osqp_vars, value=x0)
    prob.add_var(var)

    quad_obj = BoundExpr(QuadExpr(Q, q, np.zeros((1, 1))), var)
    prob.add_obj_expr(quad_obj)
    nonquad_obj = BoundExpr(Expr(f), var)

    prob.add_obj_expr(nonquad_obj)

    cnts = []
    lin_ineq = LEqExpr(AffExpr(A_ineq, -b_ineq), np.zeros(b_ineq.shape))
    lin_ineq = BoundExpr(lin_ineq, var)
    cnts.append(lin_ineq)

    nonlin_ineq = LEqExpr(Expr(g), np.zeros(g(np.zeros((2, 1))).shape))
    nonlin_ineq = BoundExpr(nonlin_ineq, var)
    cnts.append(nonlin_ineq)

    nonlin_eq = EqExpr(Expr(h), np.zeros(g(np.zeros((2, 1))).shape))
    nonlin_eq = BoundExpr(nonlin_eq, var)
    cnts.append(nonlin_eq)

    for cnt in cnts:
        prob.add_cnt_expr(cnt)

    solv.solve(prob, method="penalty_sqp")
    x_sol = var.get_value()

    ut.assertTrue(np.allclose(x_sol, x_true, atol=1e-4))


class TestSolver(unittest.TestCase):
    def test_prob0(self):
        x0 = np.array([[1.0], [1.0]])
        f = lambda x: np.array([[x[0, 0] ** 2 + x[1, 0] ** 2]])
        g = lambda x: np.array([[3 - x[0, 0] - x[1, 0]]])
        x_true = np.array([[1.5], [1.5]])
        helper_test_prob(self, x0, x_true, f=f, g=g)

    def test_prob1(self):
        x0 = np.array([[-2.0], [1.0]])
        f = lambda x: np.array([[(x[1, 0] - x[0, 0] ** 2) ** 2 + (1 - x[0, 0]) ** 2]])
        g = lambda x: np.array([[-1.5 - x[1, 0]]])
        x_true = np.array([[1.0], [1.0]])
        helper_test_prob(self, x0, x_true, f=f, g=g)

    def test_prob2(self):
        x0 = np.array([[10.0], [1.0]])
        f = lambda x: np.array([[x[1, 0] + 1e-5 + (x[1, 0] - x[0, 0]) ** 2]])
        g = lambda x: np.array([[-x[1, 0]]])
        x_true = np.array([[0.0], [0.0]])
        helper_test_prob(self, x0, x_true, f=f, g=g)

    def test_prob3(self):
        x0 = np.array([[10.0], [1.0]])
        f = lambda x: np.array([[(1 - x[0, 0]) ** 2]])
        h = lambda x: np.array([[10 * (x[1, 0] - x[0, 0] ** 2)]])
        x_true = np.array([[1.0], [1.0]])
        helper_test_prob(self, x0, x_true, f=f, h=h)

    def test_prob4(self):
        x0 = np.array([[2.0], [2.0]])
        f = lambda x: np.array([[np.log(1 + x[0, 0] ** 2) - x[1, 0]]])
        h = lambda x: np.array([[(1 + x[0, 0] ** 2) ** 2 + x[1, 0] ** 2 - 4]])
        x_true = np.array([[0.0], [np.sqrt(3)]])
        helper_test_prob(self, x0, x_true, f=f, h=h)

    def test_prob5(self):
        x0 = np.array([[0.0], [0.0]])
        angles = np.transpose(np.array(list(range(1, 7))) * 2 * np.pi / 6)
        angles = angles.reshape((6, 1))
        A_ineq = np.hstack((np.cos(angles), np.sin(angles)))
        b_ineq = np.ones(angles.shape)
        q = -np.array([[np.cos(np.pi / 6), np.sin(np.pi / 6)]])
        x_true = np.array([[1], [np.tan(np.pi / 6)]])
        helper_test_prob(self, x0, x_true, q=q, A_ineq=A_ineq, b_ineq=b_ineq)

    def test_prob6(self):
        x0 = np.array([[0.0], [0.0]])
        angles = np.transpose(np.array(list(range(1, 7))) * 2 * np.pi / 6)
        angles = angles.reshape((6, 1))
        Q = 0.1 * np.identity(2)
        A_ineq = np.hstack((np.cos(angles), np.sin(angles)))
        b_ineq = np.ones(angles.shape)
        q = -np.array([[np.cos(np.pi / 6), np.sin(np.pi / 6)]])
        g = lambda x: 0.01 * (A_ineq.dot(x) - b_ineq)
        x_true = np.transpose(np.array([[1, np.tan(np.pi / 6)]]))
        helper_test_prob(self, x0, x_true, Q=Q, q=q, g=g)

    def test_prob7(self):
        x0 = np.array([[0.0], [0.0]])
        f = lambda x: np.array([[x[0, 0] ** 4 + x[1, 0] ** 4]])
        g = lambda x: np.array([[3 - x[0, 0] - x[1, 0]]])
        h = lambda x: np.array([[x[0, 0] - 2 * x[1, 0]]])
        x_true = np.array([[2.0], [1.0]])
        helper_test_prob(self, x0, x_true, f=f, g=g, h=h)

    def test_prob8(self):
        x0 = np.array([[5.0], [5.0]])
        g = lambda x: np.vstack(
            (
                x[0, 0] ** 2 + x[1, 0] ** 2 - 4,
                -((x[0, 0] - 1) ** 2 + (x[1, 0] - 1) ** 2 - 0.25),
                -((x[0, 0] + 1) ** 2 + (x[1, 0] - 1) ** 2 - 0.25),
                -((x[0, 0]) ** 2 + 7 * (x[1, 0] + 1 - x[0, 0] ** 2 / 2) ** 2 - 0.8),
            )
        )

        Q = np.identity(2)
        x_true = np.array([[0.0], [0.0]])
        helper_test_prob(self, x0, x_true, g=g, Q=Q)
