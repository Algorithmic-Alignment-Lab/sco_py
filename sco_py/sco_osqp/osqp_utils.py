from typing import List

import numpy as np
import osqp
import scipy

from sco_py.sco_osqp.variable import Variable


DEFAULT_MAX_ITER = int(1e05)

class OSQPVar(object):
    """
    A class representing a variable within OSQP that will be used to construct the
    QP that will be passed to the solver. The lb and ub are used to enforce a
    trust region on each variable
    """

    def __init__(self, var_name, lb=-np.inf, ub=np.inf, val=None):
        self.var_name = var_name
        self._lower_bound = lb
        self._upper_bound = ub
        self.val = val

    def __lt__(self, other_osqp_var):
        return self.var_name < other_osqp_var.var_name

    def __repr__(self):
        return f"OSQPVar with name {self.var_name}"
        # return f"OSQPVar with name {self.var_name}, lb={self._lower_bound}, ub={self._upper_bound}, val={self.val}"

    def get_lower_bound(self):
        return self._lower_bound

    def set_lower_bound(self, lb_val):
        assert isinstance(lb_val, float)
        assert not np.isnan(lb_val)
        self._lower_bound = lb_val

    def get_upper_bound(self):
        return self._upper_bound

    def set_upper_bound(self, ub_val):
        assert isinstance(ub_val, float)
        assert not np.isnan(ub_val)
        self._upper_bound = ub_val


class OSQPLinearObj(object):
    def __init__(self, osqp_var, coeff):
        """
        A wrapper class for a linear objective. The particular constraint represented is:

        coeff * osqp_var
        """
        self.osqp_var = osqp_var
        self.coeff = coeff

    def __repr__(self):
        return f"OSQPLinearObj with osqp_var={self.osqp_var}, coeff={self.coeff}"

    def get_all_vars(self):
        return [self.osqp_var]


class OSQPQuadraticObj(object):
    def __init__(self, osqp_vars1, osqp_vars2, coeffs):
        """
        A wrapper class for a quadratic objective. The particular objective represented is:

        0.5 * np.linalg.multi_dot(self.osqp_vars1, self.osqp_vars2, self.coeffs)
        """
        assert osqp_vars1.shape == osqp_vars2.shape == coeffs.shape
        # Together with the above condition, this implicitly implies all these arrays must
        # have shape of length 1
        assert len(osqp_vars1.shape) == 1
        self.osqp_vars1 = osqp_vars1
        self.osqp_vars2 = osqp_vars2
        self.coeffs = coeffs

    def __repr__(self):
        return f"Quadratic Objective with osqp_vars1={self.osqp_vars1}, osqp_vars2={self.osqp_vars2}, coeffs={self.coeffs}"

    def get_all_vars(self):
        return self.osqp_vars1.tolist() + self.osqp_vars2.tolist()


class OSQPLinearConstraint(object):
    def __init__(self, osqp_vars, coeffs, lb, ub):
        """
        A wrapper class for a linear constraint. The particular constraint represented is:

        lb <= np.dot(self.osqp_vars, self.coeffs) <= ub
        """
        assert osqp_vars.shape == coeffs.shape
        self.osqp_vars = osqp_vars
        self.coeffs = coeffs
        self.lb = lb
        self.ub = ub

    def __repr__(self):
        return f"OSQPLinearConstraint with osqp_vars={self.osqp_vars}, coeffs={self.coeffs}, lb = {self.lb}, ub = {self.ub}"

    def get_all_vars(self):
        return self.osqp_vars.tolist()

# @profile
def optimize(
    osqp_vars: List[OSQPVar],
    _sco_vars: List[Variable],
    osqp_quad_objs: List[OSQPQuadraticObj],
    osqp_lin_objs: List[OSQPLinearObj],
    osqp_lin_cnt_exprs: List[OSQPLinearConstraint],
    eps_abs: float = 1e-06,
    eps_rel: float = 1e-09,
    max_iter: int = DEFAULT_MAX_ITER,
    rho: float = 1e-01,
    adaptive_rho: bool = True,
    verbose: bool = False,
):
    """
    Calls the OSQP optimizer on the current QP approximation with a given
    penalty coefficient.
    """

    # First, we need to setup the problem as described here: https://osqp.org/docs/solver/index.html
    # Specifically, we need to start by constructing the x vector that contains all the
    # OSQPVars that are part of the QP. This will take the form of a mapping from OSQPVar to
    # index within the x vector.
    var_to_index_dict = {}
    osqp_var_list = list(osqp_vars)
    # Make sure to sort this list to get a canonical ordering of variables to make 
    # matrix construction easier
    osqp_var_list.sort()  
    for idx, osqp_var in enumerate(osqp_var_list):
        var_to_index_dict[osqp_var] = idx
    num_osqp_vars = len(osqp_vars)

    # Construct the q-vector by looping through all the linear objectives
    q_vec = np.zeros(num_osqp_vars)
    for lin_obj in osqp_lin_objs:
        q_vec[var_to_index_dict[lin_obj.osqp_var]] += lin_obj.coeff

    # Next, construct the P-matrix by looping through all quadratic objectives

    # Since P must be upper-triangular, the shape must be (num_osqp_vars, num_osqp_vars)
    P_mat = np.zeros((num_osqp_vars, num_osqp_vars))
    for quad_obj in osqp_quad_objs:
        for i in range(quad_obj.coeffs.shape[0]):
            idx2 = var_to_index_dict[quad_obj.osqp_vars1[i]]
            idx1 = var_to_index_dict[quad_obj.osqp_vars2[i]]
            if idx1 > idx2:
                P_mat[idx2, idx1] += 0.5 * quad_obj.coeffs[i]
            elif idx1 < idx2:
                P_mat[idx1, idx2] += 0.5 * quad_obj.coeffs[i]
            else:
                P_mat[idx1, idx2] += quad_obj.coeffs[i]

    # Next, setup the A-matrix and l and u vectors
    A_mat = np.zeros((num_osqp_vars + len(osqp_lin_cnt_exprs), num_osqp_vars))
    l_vec = np.zeros(num_osqp_vars + len(osqp_lin_cnt_exprs))
    u_vec = np.zeros(num_osqp_vars + len(osqp_lin_cnt_exprs))

    # First add all the linear constraints
    # However, note that this isn't entirely straightforward: some 
    row_num = 0
    for lin_constraint in osqp_lin_cnt_exprs:
        l_vec[row_num] = lin_constraint.lb
        u_vec[row_num] = lin_constraint.ub
        for i in range(lin_constraint.coeffs.shape[0]):
            # if var_to_index_dict[lin_constraint.osqp_vars[i]] == 193:
            #     import ipdb; ipdb.set_trace()
            A_mat[
                row_num, var_to_index_dict[lin_constraint.osqp_vars[i]]
            ] = lin_constraint.coeffs[i]
        row_num += 1

    # Then, add the trust regions for every variable as constraints
    for osqp_var in osqp_vars:
        A_mat[row_num, var_to_index_dict[osqp_var]] = 1.0
        l_vec[row_num] = osqp_var.get_lower_bound()
        u_vec[row_num] = osqp_var.get_upper_bound()
        row_num += 1

    # Finally, construct the matrices and call the OSQP Solver!
    P_mat_sparse = scipy.sparse.csc_matrix(P_mat)
    A_mat_sparse = scipy.sparse.csc_matrix(A_mat)

    m = osqp.OSQP()

    m.setup(
        P=P_mat_sparse,
        q=q_vec,
        A=A_mat_sparse,
        rho=1e2,#rho,
        sigma=5e-10,
        l=l_vec,
        u=u_vec,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        delta=1e-07,
        # polish=True,
        polish=False,
        adaptive_rho=adaptive_rho,
        warm_start=True,
        verbose=False,
        max_iter=max_iter,
    )

    solve_res = m.solve()

    if solve_res.info.status_val == -2 and verbose:
        print("ERROR! OSQP Solver hit max iteration limit. Either reduce your tolerances or increase the max iterations!")

    return (solve_res, var_to_index_dict)


def update_osqp_vars(var_to_osqp_indices_dict, solver_values):
    """
    Updates the variables values based on the OSQP solution
    """
    for osqp_var in var_to_osqp_indices_dict.keys():
        osqp_var.val = solver_values[var_to_osqp_indices_dict[osqp_var]]

def print_osqp_vars_and_sol(solve_res_x, var_to_index_dict):
    for key, idx in var_to_index_dict.items():
        print(f"{key}, {solve_res_x[idx]}")
