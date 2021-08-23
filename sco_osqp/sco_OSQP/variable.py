import numpy as np


class Variable(object):
    """
    Variable

    Manages low-level variables by maintaining an ordering of these variables,
    """

    def __init__(self, osqp_vars, value=None):
        """
        _osqp_vars: Numpy array of OSQPVars that represent the variables
        _value: Numpy array of the current values of these variables
        _saved_value: saved value of this variable
        """
        assert isinstance(osqp_vars, np.ndarray)
        assert len(osqp_vars) > 0
        self._osqp_vars = osqp_vars.copy()
        if value is not None:
            assert osqp_vars.shape == value.shape
            assert isinstance(value, np.ndarray)
            self._value = value.copy()
        else:
            self._value = None
        self._saved_value = None

    def get_osqp_vars(self):
        return self._osqp_vars

    def get_value(self):
        if self._value is not None:
            return self._value.copy()
        else:
            return None

    def add_trust_region(self, trust_box_size):
        """
        Adds a trust region around the saved value (self._saved_value) by
        changing the upper and lower bounds of the underlying OSQP variables
        """
        assert self._saved_value is not None
        for index, osqp_var in np.ndenumerate(self._osqp_vars):
            osqp_var.set_lower_bound(self._saved_value[index] - trust_box_size)
            osqp_var.set_upper_bound(self._saved_value[index] + trust_box_size)

    def update(self):
        """
        If the OSQP variables have valid values, update self._value to reflect
        the values in these variables.
        """
        value = np.zeros(self._osqp_vars.shape)
        for index, osqp_var in np.ndenumerate(self._osqp_vars):
            if osqp_var.val is not None:
                value[index] = osqp_var.val
            else:
                raise ValueError(
                    f"The variable {osqp_var.var_name} does not have a legitimate value"
                )
        self._value = value

    def save(self):
        """
        Save the current value.
        """
        assert not np.any(np.isnan(self._value.copy()))
        self._saved_value = self._value.copy()

    def restore(self):
        """
        Restore value to the saved value.
        """
        self._value = self._saved_value.copy()
