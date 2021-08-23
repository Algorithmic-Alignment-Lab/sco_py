# fmt: off
import unittest

import numpy as np

from sco_osqp.expr import (AbsExpr, AffExpr, BoundExpr, CompExpr, EqExpr, Expr,
                           HingeExpr, LEqExpr, QuadExpr)
from sco_osqp.osqp_utils import OSQPLinearConstraint, OSQPVar
from sco_osqp.prob import Prob
from sco_osqp.variable import Variable

# fmt: on

f = lambda x: np.array([[x]])


class TestProb(unittest.TestCase):
    def test_add_obj_expr_quad(self):
        quad = QuadExpr(2 * np.eye(1), -2 * np.ones((1, 1)), np.zeros((1, 1)))
        aff = AffExpr(-2 * np.ones((1, 1)), np.zeros((1, 1)))
        prob = Prob()
        osqp_var = OSQPVar("x")
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)

        bexpr_quad = BoundExpr(quad, var)
        bexpr_aff = BoundExpr(aff, var)
        prob.add_obj_expr(bexpr_quad)
        prob.add_obj_expr(bexpr_aff)

        self.assertTrue(bexpr_aff in prob._quad_obj_exprs)
        self.assertTrue(bexpr_quad in prob._quad_obj_exprs)
        self.assertTrue(var in prob._vars)

    def test_add_obj_expr_nonquad(self):
        expr = Expr(f)
        prob = Prob()
        osqp_var = OSQPVar("x")
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)

        bexpr = BoundExpr(expr, var)
        prob.add_obj_expr(bexpr)

        self.assertTrue(bexpr not in prob._quad_obj_exprs)
        self.assertTrue(bexpr in prob._nonquad_obj_exprs)
        self.assertTrue(var in prob._vars)

    def test_find_closest_feasible_point_leq_cnts(self):
        cnt_vals = [
            np.ones((2, 1)),
            np.array([[-1.0], [1.0]]),
            np.array([[-1.0], [-1.0]]),
        ]
        true_var_vals = [
            np.zeros((2, 1)),
            np.array([[-1.0], [0.0]]),
            -1 * np.ones((2, 1)),
        ]

        for true_var_val, cnt_val in zip(true_var_vals, cnt_vals):
            prob = Prob()
            osqp_var1 = OSQPVar("x1")
            osqp_var2 = OSQPVar("x2")
            prob.add_osqp_var(osqp_var1)
            prob.add_osqp_var(osqp_var2)
            osqp_vars = np.array([[osqp_var1], [osqp_var2]])
            var = Variable(osqp_vars, np.zeros((2, 1)))
            prob.add_var(var)

            aff_expr = AffExpr(np.eye(2), np.zeros((2, 1)))
            leq_expr = LEqExpr(aff_expr, cnt_val)
            bexpr = BoundExpr(leq_expr, var)
            prob.add_cnt_expr(bexpr)
            prob.find_closest_feasible_point()
            self.assertTrue(np.allclose(var.get_value(), true_var_val))

    def test_find_closest_feasible_point_eq_cnts(self):
        prob = Prob()
        osqp_var1 = OSQPVar("x1")
        osqp_var2 = OSQPVar("x2")
        prob.add_osqp_var(osqp_var1)
        prob.add_osqp_var(osqp_var2)
        osqp_vars = np.array([[osqp_var1], [osqp_var2]])
        var = Variable(osqp_vars, np.zeros((2, 1)))
        prob.add_var(var)

        val = np.array([[5.0], [-10.0]])
        aff_expr = AffExpr(np.eye(2), np.zeros((2, 1)))
        eq_expr = EqExpr(aff_expr, val)
        bexpr = BoundExpr(eq_expr, var)
        prob.add_cnt_expr(bexpr)
        prob.find_closest_feasible_point()
        self.assertTrue(np.allclose(var.get_value(), val))

    def test_optimize_just_quad_obj(self):
        quad = QuadExpr(2 * np.eye(1), -2 * np.ones((1, 1)), np.zeros((1, 1)))
        aff = AffExpr(-2 * np.ones((1, 1)), np.zeros((1, 1)))
        prob = Prob()
        osqp_var = OSQPVar("x")
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        bexpr_quad = BoundExpr(quad, var)
        bexpr_aff = BoundExpr(aff, var)
        prob.add_obj_expr(bexpr_quad)
        prob.add_obj_expr(bexpr_aff)

        self.assertTrue(bexpr_aff in prob._quad_obj_exprs)
        self.assertTrue(bexpr_quad in prob._quad_obj_exprs)
        self.assertTrue(var in prob._vars)

        prob.update_obj(penalty_coeff=0)
        prob.optimize()

        self.assertTrue(np.allclose(var.get_value(), np.array([[1.0]])))

    def test_optimize_multidim_quad_obj(self):
        Q = np.array([[2, 0], [0, 0]])
        A = np.array([[-4, 0]])
        quad = QuadExpr(Q, A, np.zeros((1, 1)))
        prob = Prob()
        osqp_var1 = OSQPVar("x1")
        osqp_var2 = OSQPVar("x2")
        prob.add_osqp_var(osqp_var1)
        prob.add_osqp_var(osqp_var2)
        osqp_vars = np.array([[osqp_var1], [osqp_var2]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        bexpr_quad = BoundExpr(quad, var)
        prob.add_obj_expr(bexpr_quad)

        self.assertTrue(bexpr_quad in prob._quad_obj_exprs)
        self.assertTrue(var in prob._vars)

        prob.update_obj(penalty_coeff=0)
        prob.optimize()
        var_value = var.get_value()
        value = np.zeros((2, 1))
        value[0, 0] = 2

        self.assertTrue(np.allclose(var_value, value))

    def test_expr_to_osqp_expr_w_comp_expr(self):
        prob = Prob()
        aff = AffExpr(-2 * np.ones((1, 1)), np.zeros((1, 1)))
        val = np.zeros((1, 1))
        cexpr = CompExpr(aff, val)
        bexpr = BoundExpr(cexpr, None)
        with self.assertRaises(Exception) as e:
            prob._add_osqp_objs_and_cnts_from_expr(bexpr)
        self.assertTrue("Comparison" in str(e.exception))

    def test_expr_to_osqp_expr_w_expr(self):
        prob = Prob()
        bexpr = BoundExpr(f, None)
        with self.assertRaises(Exception) as e:
            prob._add_osqp_objs_and_cnts_from_expr(bexpr)
        self.assertTrue("Expression cannot be converted" in str(e.exception))

    def test_add_cnt_expr_eq_aff(self):
        aff = AffExpr(np.ones((1, 1)), np.zeros((1, 1)))
        comp = EqExpr(aff, np.array([[2]]))
        prob = Prob()
        osqp_var = OSQPVar("x")
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        bexpr = BoundExpr(comp, var)
        prob.add_cnt_expr(bexpr)

        prob.optimize()
        self.assertTrue(np.allclose(var.get_value(), np.array([[2]])))

    def test_add_cnt_leq_aff(self):
        """
        minimize x^2 - x st. x <= -4
        """
        quad = QuadExpr(2 * np.eye(1), -2 * np.ones((1, 1)), np.zeros((1, 1)))

        prob = Prob()
        osqp_var = OSQPVar("x", ub=-4.0)
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        bexpr_quad = BoundExpr(quad, var)
        prob.add_obj_expr(bexpr_quad)
        prob.update_obj()

        prob.optimize(add_convexified_terms=True)

        self.assertTrue(np.allclose(var.get_value(), np.array([[-4]])))

    def test_hinge_expr_to_osqp_expr1(self):
        """
        min max(0, x+1) st. x == -4
        """
        aff = AffExpr(np.ones((1, 1)), np.ones((1, 1)))
        hinge = HingeExpr(aff)

        prob = Prob()
        osqp_var = OSQPVar("x")
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        prob._add_to_lin_objs_and_cnts_from_hinge_expr(hinge, var)

        aff = AffExpr(np.ones((1, 1)), np.zeros((1, 1)))
        comp = EqExpr(aff, np.array([[-4]]))
        bound_expr = BoundExpr(comp, var)
        prob.add_cnt_expr(bound_expr)

        prob.optimize()
        var.update()
        self.assertTrue(np.allclose(var.get_value(), np.array([[-4]])))

    def test_hinge_expr_to_osqp_expr2(self):
        """
        min max(0, x+1) st. x == 1
        """
        aff = AffExpr(np.ones((1, 1)), np.ones((1, 1)))
        hinge = HingeExpr(aff)

        prob = Prob()
        osqp_var = OSQPVar("x")
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        prob._add_to_lin_objs_and_cnts_from_hinge_expr(hinge, var)

        aff = AffExpr(np.ones((1, 1)), np.zeros((1, 1)))
        comp = EqExpr(aff, np.array([[1.0]]))
        bound_expr = BoundExpr(comp, var)
        prob.add_cnt_expr(bound_expr)

        prob.optimize()
        var.update()
        self.assertTrue(np.allclose(var.get_value(), np.array([[1.0]])))

    def test_abs_expr_to_osqp_expr(self):
        """
        min |x + 1| s.t. x <= -4
        """
        aff = AffExpr(np.ones((1, 1)), np.ones((1, 1)))
        abs_expr = AbsExpr(aff)

        prob = Prob()
        osqp_var = OSQPVar("x", ub=-4.0)
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        prob._add_to_lin_objs_and_cnts_from_abs_expr(abs_expr, var)

        prob.optimize(add_convexified_terms=True)
        var.update()
        self.assertTrue(np.allclose(var.get_value(), np.array([[-4]])))

    def test_convexify_eq(self):
        prob = Prob()
        osqp_var = OSQPVar("x")
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        prob._osqp_lin_cnt_exprs += [
            OSQPLinearConstraint(np.array([osqp_var]), np.ones(1), 0.0, 0.0)
        ]
        prob.optimize()
        var.update()

        e = Expr(f)
        eq = EqExpr(e, np.array([[4]]))
        bexpr = BoundExpr(eq, var)
        prob.add_cnt_expr(bexpr)

        prob.convexify()
        self.assertTrue(len(prob._penalty_exprs) == 1)
        self.assertTrue(isinstance(prob._penalty_exprs[0].expr, AbsExpr))

    def test_convexify_leq(self):
        prob = Prob()
        osqp_var = OSQPVar("x")
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        prob._osqp_lin_cnt_exprs += [
            OSQPLinearConstraint(np.array([osqp_var]), np.ones(1), 0.0, 0.0)
        ]
        prob.optimize()
        var.update()

        e = Expr(f)
        eq = LEqExpr(e, np.array([[4]]))
        bexpr = BoundExpr(eq, var)
        prob.add_cnt_expr(bexpr)

        prob.convexify()
        self.assertTrue(len(prob._penalty_exprs) == 1)
        self.assertTrue(isinstance(prob._penalty_exprs[0].expr, HingeExpr))

    def test_get_value_lin_constr(self):
        """
        min x^2 st. x == 4
        when convexified,
        min x^2 + penalty_coeff*|x-4|
        when penalty_coeff == 1, solution is x = 0.5 and the value is 3.75
        (according to Wolfram Alpha)

        when penalty_coeff == 2, solution is x = 1.0 and the value is 7.0
        (according to Wolfram Alpha)
        """
        quad = QuadExpr(2 * np.eye(1), np.zeros((1, 1)), np.zeros((1, 1)))
        e = Expr(f)
        eq = EqExpr(e, np.array([[4]]))

        prob = Prob()

        osqp_var = OSQPVar("x")
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        obj = BoundExpr(quad, var)
        prob.add_obj_expr(obj)
        bexpr = BoundExpr(eq, var)
        prob.add_cnt_expr(bexpr)

        prob.optimize()  # needed to set an initial value
        prob.convexify()
        prob.update_obj(penalty_coeff=1.0)
        prob.optimize()

        self.assertTrue(np.allclose(var.get_value(), np.array([[0.5]])))
        self.assertTrue(np.allclose(prob.get_value(1.0), np.array([[3.75]])))

        prob.update_obj(penalty_coeff=2.0)
        prob.optimize()
        self.assertTrue(np.allclose(var.get_value(), np.array([[1.0]])))
        self.assertTrue(np.allclose(prob.get_value(2.0), np.array([[7]])))

    def test_get_approx_value_lin_constr(self):
        """
        min x^2 st. x == 4
        when convexified,
        min x^2 + penalty_coeff*|x-4|
        when penalty_coeff == 1, solution is x = 0.5 and the value is 3.75
        (according to Wolfram Alpha)

        when penalty_coeff == 2, solution is x = 1.0 and the value is 7.0
        (according to Wolfram Alpha)
        """
        quad = QuadExpr(2 * np.eye(1), np.zeros((1, 1)), np.zeros((1, 1)))
        e = Expr(f)
        eq = EqExpr(e, np.array([[4]]))

        prob = Prob()

        osqp_var = OSQPVar("x")
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        prob.add_var(var)

        obj = BoundExpr(quad, var)
        prob.add_obj_expr(obj)
        bexpr = BoundExpr(eq, var)
        prob.add_cnt_expr(bexpr)

        prob.optimize()  # needed to set an initial value
        prob.convexify()
        prob.update_obj(penalty_coeff=1.0)
        prob.optimize()
        self.assertTrue(np.allclose(var.get_value(), np.array([[0.5]])))
        self.assertTrue(np.allclose(prob.get_approx_value(1.0), np.array([[3.75]])))

        prob.update_obj(penalty_coeff=2.0)
        prob.optimize()
        self.assertTrue(np.allclose(var.get_value(), np.array([[1.0]])))
        self.assertTrue(np.allclose(prob.get_approx_value(2.0), np.array([[7]])))

    def test_get_value_and_get_approx_value_nonlin_constr(self):
        """
        min x^2 -2x + 1 st. x^2 == 4
        when convexified at x = 1,
        min x^2 -2x + 1 + penalty_coeff*|2x-5|
        when penalty_coeff == 0.5, solution is x = 1.5 and the value is 1.25
        (according to Wolfram Alpha)

        approx value should be 1.25
        value should be 1.125
        """
        quad = QuadExpr(2 * np.eye(1), -2 * np.ones((1, 1)), np.ones((1, 1)))
        quad_cnt = QuadExpr(2 * np.eye(1), np.zeros((1, 1)), np.zeros((1, 1)))
        eq = EqExpr(quad_cnt, np.array([[4]]))

        prob = Prob()

        osqp_var = OSQPVar("x")
        prob.add_osqp_var(osqp_var)
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars, np.array([[1.0]]))
        prob.add_var(var)

        obj = BoundExpr(quad, var)
        prob.add_obj_expr(obj)
        bexpr = BoundExpr(eq, var)
        prob.add_cnt_expr(bexpr)

        prob.convexify()
        prob.update_obj(penalty_coeff=0.5)
        prob.optimize()

        self.assertTrue(np.allclose(var.get_value(), np.array([[1.5]])))
        self.assertTrue(np.allclose(prob.get_approx_value(0.5), np.array([[1.25]])))
        self.assertTrue(np.allclose(prob.get_value(0.5), np.array([[1.125]])))

    def test_get_max_cnt_violation_eq_cnts(self):
        prob = Prob()
        dummy_var = Variable(np.zeros((1, 1)), np.zeros((1, 1)))
        f = lambda x: np.array([[1, 3]])

        f_expr = Expr(f)
        eq_expr = EqExpr(f_expr, np.array([[1, 1]]))
        bexpr = BoundExpr(eq_expr, dummy_var)
        prob.add_cnt_expr(bexpr)
        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 2.0))

        f_expr = Expr(f)
        f_expr.f = lambda x: np.array([[2, 1]])
        eq_expr.expr = f_expr
        eq_expr.val = np.array([[1, 1]])
        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 1.0))

        f_expr = Expr(f)
        f_expr.f = lambda x: np.array([[2, -2]])
        eq_expr.expr = f_expr
        eq_expr.val = np.array([[1, 1]])
        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 3.0))

        f_expr = Expr(f)
        f_expr.f = lambda x: np.array([[2, -2]])
        eq_expr.expr = f_expr
        eq_expr.val = np.array([[2, -2]])
        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 0.0))

        f_expr = Expr(f)
        f_expr.f = lambda x: np.array([[2, 0]])
        eq_expr.expr = f_expr
        eq_expr.val = np.array([[2, -2]])
        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 2.0))

    def test_get_max_cnt_violation_leq_cnts(self):
        prob = Prob()
        dummy_var = Variable(np.zeros((1, 1)), np.zeros((1, 1)))
        f = lambda x: np.array([[1, 3]])

        f_expr = Expr(f)
        leq_expr = LEqExpr(f_expr, np.array([[1, 1]]))
        bexpr = BoundExpr(leq_expr, dummy_var)
        prob.add_cnt_expr(bexpr)
        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 2.0))

        f_expr = Expr(f)
        f_expr.f = lambda x: np.array([[2, 1]])
        leq_expr.expr = f_expr
        leq_expr.val = np.array([[1, 1]])
        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 1.0))

        f_expr = Expr(f)
        f_expr.f = lambda x: np.array([[2, -2]])
        leq_expr.expr = f_expr
        leq_expr.val = np.array([[1, 1]])
        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 1.0))

        f_expr = Expr(f)
        f_expr.f = lambda x: np.array([[2, -2]])
        leq_expr.expr = f_expr
        leq_expr.val = np.array([[2, -2]])
        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 0.0))

        f_expr = Expr(f)
        f_expr.f = lambda x: np.array([[2, 0]])
        leq_expr.expr = f_expr
        leq_expr.val = np.array([[2, -2]])
        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 2.0))

    def test_get_max_cnt_violation_mult_cnts(self):
        prob = Prob()
        dummy_var = Variable(np.zeros((1, 1)), np.zeros((1, 1)))
        f1 = lambda x: np.array([[1, 3]])
        f2 = lambda x: np.array([[0, 0]])

        f1_expr = Expr(f1)
        leq_expr = LEqExpr(f1_expr, np.array([[1, 1]]))
        bexpr = BoundExpr(leq_expr, dummy_var)
        prob.add_cnt_expr(bexpr)

        f2_expr = Expr(f2)
        eq_expr = EqExpr(f2_expr, np.array([[1, 1]]))
        bexpr = BoundExpr(eq_expr, dummy_var)
        prob.add_cnt_expr(bexpr)

        self.assertTrue(np.allclose(prob.get_max_cnt_violation(), 2.0))
