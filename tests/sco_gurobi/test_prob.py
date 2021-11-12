# fmt: off
import unittest

import gurobipy as grb
import numpy as np

from sco.expr import (AbsExpr, AffExpr, BoundExpr, CompExpr, EqExpr,
                             Expr, HingeExpr, LEqExpr, QuadExpr)
from sco.sco_gurobi.prob import PosGRBVarManager, Prob
from sco.sco_gurobi.variable import Variable

# fmt: on

GRB = grb.GRB


f = lambda x: np.array([[x]])


def helper_test_grb_var_pos(ut, grb_var):
    ut.assertTrue(grb_var.lb == 0.0 and grb_var.ub == np.inf)


class TestProb(unittest.TestCase):
    def test_add_obj_expr_quad(self):
        quad = QuadExpr(2 * np.eye(1), -2 * np.ones((1, 1)), np.zeros((1, 1)))
        aff = AffExpr(-2 * np.ones((1, 1)), np.zeros((1, 1)))
        model = grb.Model()
        prob = Prob(model)
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        bexpr_quad = BoundExpr(quad, var)
        bexpr_aff = BoundExpr(aff, var)
        prob.add_obj_expr(bexpr_quad)
        prob.add_obj_expr(bexpr_aff)

        self.assertTrue(bexpr_aff in prob._quad_obj_exprs)
        self.assertTrue(bexpr_quad in prob._quad_obj_exprs)
        self.assertTrue(var in prob._vars)

    def test_add_obj_expr_nonquad(self):
        expr = Expr(f)
        model = grb.Model()
        prob = Prob(model)
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

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
            model = grb.Model()
            prob = Prob(model)
            grb_var1 = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x1")
            grb_var2 = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x2")
            grb_vars = np.array([[grb_var1], [grb_var2]])
            var = Variable(grb_vars, np.zeros((2, 1)))

            model.update()

            aff_expr = AffExpr(np.eye(2), np.zeros((2, 1)))
            leq_expr = LEqExpr(aff_expr, cnt_val)
            bexpr = BoundExpr(leq_expr, var)
            prob.add_cnt_expr(bexpr)
            prob.find_closest_feasible_point()
            self.assertTrue(np.allclose(var.get_value(), true_var_val))

    def test_find_closest_feasible_point_eq_cnts(self):
        model = grb.Model()
        prob = Prob(model)
        grb_var1 = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x1")
        grb_var2 = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x2")
        grb_vars = np.array([[grb_var1], [grb_var2]])
        var = Variable(grb_vars, np.zeros((2, 1)))

        model.update()

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
        model = grb.Model()
        prob = Prob(model)
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        bexpr_quad = BoundExpr(quad, var)
        bexpr_aff = BoundExpr(aff, var)
        prob.add_obj_expr(bexpr_quad)
        prob.add_obj_expr(bexpr_aff)

        self.assertTrue(bexpr_aff in prob._quad_obj_exprs)
        self.assertTrue(bexpr_quad in prob._quad_obj_exprs)
        self.assertTrue(var in prob._vars)

        prob.update_obj(penalty_coeff=0)
        prob.optimize()
        self.assertTrue(np.allclose(var.get_value(), np.array([[2.0]])))

    def test_optimize_multidim_quad_obj(self):
        Q = np.array([[2, 0], [0, 0]])
        A = np.array([[-4, 0]])
        quad = QuadExpr(Q, A, np.zeros((1, 1)))
        model = grb.Model()
        prob = Prob(model)
        grb_var1 = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x1")
        grb_var2 = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x2")
        grb_vars = np.array([[grb_var1], [grb_var2]])
        var = Variable(grb_vars)

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

    def test_expr_to_grb_expr_w_comp_expr(self):
        model = grb.Model()
        prob = Prob(model)
        aff = AffExpr(-2 * np.ones((1, 1)), np.zeros((1, 1)))
        val = np.zeros((1, 1))
        cexpr = CompExpr(aff, val)
        bexpr = BoundExpr(cexpr, None)
        try:
            prob._expr_to_grb_expr(bexpr)
        except Exception as e:
            self.assertTrue("Comparison" in str(e))

    def test_expr_to_grb_expr_w_expr(self):
        model = grb.Model()
        prob = Prob(model)
        bexpr = BoundExpr(f, None)
        try:
            prob._expr_to_grb_expr(bexpr)
        except Exception as e:
            self.assertTrue("Expression cannot be converted" in str(e))

    def test_add_cnt_expr_eq_aff(self):
        aff = AffExpr(np.ones((1, 1)), np.zeros((1, 1)))
        comp = EqExpr(aff, np.array([[2]]))
        model = grb.Model()
        prob = Prob(model)
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        bexpr = BoundExpr(comp, var)
        prob.add_cnt_expr(bexpr)

        prob.optimize()
        self.assertTrue(np.allclose(var.get_value(), np.array([[2]])))

    def test_add_cnt_leq_aff(self):
        quad = QuadExpr(2 * np.eye(1), -2 * np.ones((1, 1)), np.zeros((1, 1)))

        aff = AffExpr(np.ones((1, 1)), np.zeros((1, 1)))
        comp = LEqExpr(aff, np.array([[-4]]))
        model = grb.Model()
        prob = Prob(model)
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        bexpr_quad = BoundExpr(quad, var)
        prob.add_obj_expr(bexpr_quad)

        bexpr = BoundExpr(comp, var)
        prob.add_cnt_expr(bexpr)

        prob.optimize()
        self.assertTrue(np.allclose(var.get_value(), np.array([[-4]])))

    def test_hinge_expr_to_grb_expr1(self):
        """
        min max(0, x+1) st. x == -4
        """
        aff = AffExpr(np.ones((1, 1)), np.ones((1, 1)))
        hinge = HingeExpr(aff)
        model = grb.Model()
        prob = Prob(model)

        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        hinge_grb_expr, hinge_grb_cnt = prob._hinge_expr_to_grb_expr(hinge, var)
        model.update()
        obj = hinge_grb_expr[0, 0]
        model.setObjective(obj)

        aff = AffExpr(np.ones((1, 1)), np.zeros((1, 1)))
        comp = EqExpr(aff, np.array([[-4]]))
        bound_expr = BoundExpr(comp, var)
        prob.add_cnt_expr(bound_expr)

        model.optimize()
        var.update()
        self.assertTrue(np.allclose(var.get_value(), np.array([[-4]])))
        self.assertTrue(np.allclose(obj.X, 0.0))

    def test_hinge_expr_to_grb_expr2(self):
        """
        min max(0, x+1) st. x == 1
        """
        aff = AffExpr(np.ones((1, 1)), np.ones((1, 1)))
        hinge = HingeExpr(aff)
        model = grb.Model()
        prob = Prob(model)

        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

        hinge_grb_expr, hinge_grb_cnt = prob._hinge_expr_to_grb_expr(hinge, var)
        model.update()
        obj = hinge_grb_expr[0, 0]
        model.setObjective(obj)

        aff = AffExpr(np.ones((1, 1)), np.zeros((1, 1)))
        comp = EqExpr(aff, np.array([[1.0]]))
        bound_expr = BoundExpr(comp, var)
        prob.add_cnt_expr(bound_expr)

        model.optimize()
        var.update()
        self.assertTrue(np.allclose(var.get_value(), np.array([[1.0]])))
        self.assertTrue(np.allclose(obj.X, 2.0))

    def test_abs_expr_to_grb_expr(self):
        """
        min |x + 1| s.t. x <= -4
        """
        aff = AffExpr(np.ones((1, 1)), np.ones((1, 1)))
        abs_expr = AbsExpr(aff)

        aff = AffExpr(np.ones((1, 1)), np.zeros((1, 1)))
        comp = LEqExpr(aff, np.array([[-4]]))

        model = grb.Model()
        prob = Prob(model)

        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        model.update()

        abs_grb_expr, abs_grb_cnt = prob._abs_expr_to_grb_expr(abs_expr, var)
        model.update()
        model.setObjective(abs_grb_expr[0, 0])

        bexpr = BoundExpr(comp, var)
        prob.add_cnt_expr(bexpr)

        model.optimize()
        var.update()

        self.assertTrue(np.allclose(var.get_value(), np.array([[-4]])))
        # makes assumption about the construction of the Gurobi variable, needs
        # to be changed TODO
        pos = abs_grb_expr[0, 0].getVar(0).X
        neg = abs_grb_expr[0, 0].getVar(1).X
        self.assertTrue(np.allclose(pos, 0.0))
        self.assertTrue(np.allclose(neg, 3.0))

    def test_convexify_eq(self):
        model = grb.Model()
        prob = Prob(model)
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        model.update()
        model.addConstr(grb_var, GRB.EQUAL, 0)
        model.optimize()
        var.update()

        e = Expr(f)
        eq = EqExpr(e, np.array([[4]]))
        bexpr = BoundExpr(eq, var)
        prob.add_cnt_expr(bexpr)

        prob.convexify()
        self.assertTrue(len(prob._penalty_exprs) == 1)
        self.assertTrue(isinstance(prob._penalty_exprs[0].expr, AbsExpr))

    def test_convexify_leq(self):
        model = grb.Model()
        prob = Prob(model)
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)

        model.update()
        model.addConstr(grb_var, GRB.EQUAL, 0)
        model.optimize()
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

        model = grb.Model()
        prob = Prob(model)

        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

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

        model = grb.Model()
        prob = Prob(model)

        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars)
        model.update()

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

        model = grb.Model()
        prob = Prob(model)

        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        grb_vars = np.array([[grb_var]])
        var = Variable(grb_vars, np.array([[1.0]]))
        model.update()

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
        model = grb.Model()
        prob = Prob(model)
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
        model = grb.Model()
        prob = Prob(model)
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
        model = grb.Model()
        prob = Prob(model)
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

    def test_pos_grb_var_manager(self):
        model = grb.Model()
        init_num = 1
        inc_num = 10
        shape = (2, 7)
        pos_grb_manager = PosGRBVarManager(model, init_num=init_num, inc_num=inc_num)
        pgm = pos_grb_manager
        PosGRBVarManager.INC_NUM = 2
        self.assertTrue(len(pgm._grb_vars) == init_num)
        var = next(pgm)
        helper_test_grb_var_pos(self, var)
        self.assertTrue(len(pgm._grb_vars) == inc_num + init_num)
        a = pgm.get_array(shape)
        for x in a.flatten():
            helper_test_grb_var_pos(self, x)

        self.assertTrue(a.shape == shape)
        self.assertTrue(len(pgm._grb_vars) == init_num + 2 * inc_num)

    def test_callback(self):
        x = {}

        def test():
            x[1] = 2

        callback = test
        model = grb.Model()
        prob = Prob(model, callback)

        prob.find_closest_feasible_point()
        self.assertTrue(1 in x)
        x[1] = 3
        prob.optimize()
        self.assertTrue(1 in x)
