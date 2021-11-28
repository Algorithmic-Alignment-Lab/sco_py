import unittest

import gurobipy as grb
import numpy as np

from pysco.sco_gurobi.variable import Variable

GRB = grb.GRB


class TestVariable(unittest.TestCase):
    def test_variable(self):
        ## test initialization. Ensure that
        one = np.array([1])
        var = Variable(one)
        self.assertEqual(var._grb_vars, one)
        self.assertEqual(var._value, None)
        self.assertEqual(var._saved_value, None)

        one = np.array([1])
        two = np.array([2])
        var = Variable(one, two)
        self.assertTrue(np.allclose(var._value, two))
        two[0] = 1
        self.assertFalse(np.allclose(var._value, two))

    def test_get_value(self):
        ## test that get_value returns None when self._value has no value
        ## test that get_value returns the correct value after update
        ## test to ensure that modifying returned value won't modify self._value
        ## in the Variable class
        model = grb.Model()
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        model.update()

        grb_vars = np.array([grb_var])
        var = Variable(grb_vars)
        self.assertEqual(var.get_value(), None)

        obj = grb.QuadExpr()
        obj += grb_var * grb_var - 4 * grb_var + 4

        model.setObjective(obj)
        model.params.OutputFlag = 0
        model.optimize()
        var.update()
        val = var.get_value()
        self.assertTrue(np.allclose(val, np.array([2.0])))

        val[0] = 1.0
        self.assertTrue(np.allclose(var.get_value(), np.array([2.0])))

    def test_add_trust_region(self):
        ## test trust region being computed correctly and having the correct
        ## effect on the Gurobi Optimization problem

        model = grb.Model()
        model.params.OutputFlag = 0
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        model.update()

        grb_vars = np.array([grb_var])
        var = Variable(grb_vars)
        var._saved_value = np.array([4.0])
        var.add_trust_region(1.0)

        model.update()  # upper and lower bounds aren't set until model update
        self.assertTrue(grb_var.lb == 3.0)
        self.assertTrue(grb_var.ub == 5.0)

        obj = grb.QuadExpr()
        obj += grb_var * grb_var - 4 * grb_var + 4
        model.setObjective(obj)
        model.optimize()

        var.update()
        self.assertTrue(np.allclose(var._value, np.array([3.0])))

    def test_update(self):
        ## test that update updates self._value to values in Variable's Gurobi
        ## variables and that a GurobiError is raised if Variable's Gurobi
        ## variables do not have valid values
        model = grb.Model()
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        model.update()

        grb_vars = np.array([grb_var])
        var = Variable(grb_vars)
        with self.assertRaises(AttributeError) as _:
            var.update()

        obj = grb.QuadExpr()
        obj += grb_var * grb_var - 4 * grb_var + 4
        model.setObjective(obj)
        model.params.OutputFlag = 0
        model.optimize()

        var.update()
        self.assertTrue(np.allclose(var.get_value(), np.array([2.0])))

    def test_save_and_restore(self):
        ## test to ensure saves sets self._saved_value correctly and modifying
        ## self._value doesn't change self._saved_value
        ## test to ensure that restore sets self._value to self._saved_value
        model = grb.Model()
        grb_var = model.addVar(lb=-1 * GRB.INFINITY, ub=GRB.INFINITY, name="x")
        model.update()

        grb_vars = np.array([grb_var])
        var = Variable(grb_vars)

        obj = grb.QuadExpr()
        obj += grb_var * grb_var - 4 * grb_var + 4
        model.setObjective(obj)
        model.params.OutputFlag = 0
        model.optimize()

        var.update()
        var.save()
        self.assertTrue(np.allclose(var._value, np.array([2.0])))

        obj = grb_var * grb_var - 2 * grb_var + 1
        model.setObjective(obj)
        model.optimize()
        var.update()
        self.assertTrue(np.allclose(var._value, np.array([1.0])))

        self.assertTrue(np.allclose(var._saved_value, np.array([2.0])))

        var.restore()
        self.assertTrue(np.allclose(var._value, np.array([2.0])))


if __name__ == "__main__":
    unittest.main()
