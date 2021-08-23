# fmt: off
import unittest

import numpy as np

import sco_osqp.osqp_utils as osqp_utils
from sco_osqp.osqp_utils import OSQPLinearObj, OSQPQuadraticObj, OSQPVar
from sco_osqp.variable import Variable

# fmt: on


class TestVariable(unittest.TestCase):
    def test_variable(self):
        # test initialization. Ensure that
        one = np.array([1])
        var = Variable(one)
        self.assertEqual(var._osqp_vars, one)
        self.assertEqual(var._value, None)
        self.assertEqual(var._saved_value, None)

        one = np.array([1])
        two = np.array([2])
        var = Variable(one, two)
        self.assertTrue(np.allclose(var._value, two))
        two[0] = 1
        self.assertFalse(np.allclose(var._value, two))

    def test_get_osqp_vars(self):
        # test to ensure that modifying returned OSQP Variables won't modify
        # the OSQP variables (self._osqp_vars) in the Variable class
        osqp_var = OSQPVar("x")
        osqp_vars = np.array([osqp_var])
        var = Variable(osqp_vars)
        self.assertEqual(osqp_vars, var.get_osqp_vars())
        osqp_vars[0] = 0.0
        self.assertNotEqual(osqp_vars, var.get_osqp_vars())

    def test_get_value(self):
        # test that get_value returns None when self._value has no value
        # test that get_value returns the correct value after update
        # test to ensure that modifying returned value won't modify self._value
        # in the Variable class
        osqp_var = OSQPVar("x")

        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        self.assertEqual(var.get_value(), None)

        quad_obj = OSQPQuadraticObj(
            np.array([osqp_var]), np.array([osqp_var]), np.array([2.0])
        )
        lin_obj = OSQPLinearObj(osqp_var, -4.0)

        solve_res, var_to_indices_dict = osqp_utils.optimize(
            [osqp_var], [var], [quad_obj], [lin_obj], []
        )
        assert solve_res.info.status_val in [1, 2]

        osqp_utils.update_osqp_vars(var_to_indices_dict, solve_res.x)
        var.update()
        val = var.get_value()

        self.assertTrue(np.allclose(val, np.array([2.0])))

        val[0] = 1.0
        self.assertTrue(np.allclose(var.get_value(), np.array([2.0])))

    def test_add_trust_region(self):
        # test trust region being computed correctly and having the correct
        # effect on the OSQP Optimization problem

        osqp_var = OSQPVar("x")
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        var._saved_value = np.array([[4.0]])
        var.add_trust_region(1.0)

        self.assertTrue(osqp_var.get_lower_bound() == 3.0)
        self.assertTrue(osqp_var.get_upper_bound() == 5.0)

        quad_obj = OSQPQuadraticObj(
            np.array([osqp_var]), np.array([osqp_var]), np.array([2.0])
        )
        lin_obj = OSQPLinearObj(osqp_var, -4.0)

        solve_res, var_to_indices_dict = osqp_utils.optimize(
            [osqp_var], [var], [quad_obj], [lin_obj], []
        )
        assert solve_res.info.status_val in [1, 2]

        osqp_utils.update_osqp_vars(var_to_indices_dict, solve_res.x)
        var.update()
        val = var.get_value()

        self.assertTrue(np.allclose(val, np.array([3.0])))

    def test_update(self):
        # test that update updates self._value to values in Variable's OSQP
        # variables and that an error is raised if Variable's OSQP
        # variables do not have valid values
        osqp_var = OSQPVar("x")
        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)
        with self.assertRaises(ValueError) as _:
            var.update()

        quad_obj = OSQPQuadraticObj(
            np.array([osqp_var]), np.array([osqp_var]), np.array([2.0])
        )
        lin_obj = OSQPLinearObj(osqp_var, -4.0)

        solve_res, var_to_indices_dict = osqp_utils.optimize(
            [osqp_var], [var], [quad_obj], [lin_obj], []
        )
        assert solve_res.info.status_val in [1, 2]

        osqp_utils.update_osqp_vars(var_to_indices_dict, solve_res.x)
        var.update()
        val = var.get_value()

        self.assertTrue(np.allclose(val, np.array([2.0])))

        val[0] = 1.0
        self.assertTrue(np.allclose(var.get_value(), np.array([2.0])))

    def test_save_and_restore(self):
        # test to ensure saves sets self._saved_value correctly and modifying
        # self._value doesn't change self._saved_value
        # test to ensure that restore sets self._value to self._saved_value
        osqp_var = OSQPVar("x")

        osqp_vars = np.array([[osqp_var]])
        var = Variable(osqp_vars)

        quad_obj = OSQPQuadraticObj(
            np.array([osqp_var]), np.array([osqp_var]), np.array([2.0])
        )
        lin_obj = OSQPLinearObj(osqp_var, -4.0)

        solve_res, var_to_indices_dict = osqp_utils.optimize(
            [osqp_var], [var], [quad_obj], [lin_obj], []
        )
        assert solve_res.info.status_val in [1, 2]
        osqp_utils.update_osqp_vars(var_to_indices_dict, solve_res.x)
        var.update()
        var.save()
        self.assertTrue(np.allclose(var._value, np.array([2.0])))

        quad_obj = OSQPQuadraticObj(
            np.array([osqp_var]), np.array([osqp_var]), np.array([2.0])
        )
        lin_obj = OSQPLinearObj(osqp_var, -2.0)

        solve_res, var_to_indices_dict = osqp_utils.optimize(
            [osqp_var], [var], [quad_obj], [lin_obj], []
        )
        assert solve_res.info.status_val in [1, 2]
        osqp_utils.update_osqp_vars(var_to_indices_dict, solve_res.x)
        var.update()
        self.assertTrue(np.allclose(var._value, np.array([1.0])))

        self.assertTrue(np.allclose(var._saved_value, np.array([2.0])))

        var.restore()
        self.assertTrue(np.allclose(var._value, np.array([2.0])))
