# fmt: off
from collections import defaultdict

import numpy as np

import sco_py.expr as sco_osqp_expr
import sco_py.sco_osqp.osqp_utils as osqp_utils
from sco_py.sco_osqp.osqp_utils import (OSQPLinearConstraint, OSQPLinearObj,
                                 OSQPQuadraticObj, OSQPVar)

# fmt: on


class Prob(object):
    """
    Sequential convex programming problem with a scalar objective. A solution is
    found using the l1 penalty method.
    """

    def __init__(self, callback=None):
        """
        _vars: variables in this problem
        _osqp_vars: a set of all the osqp_vars in this problem. This will be used to
            construct the x vector whenever a call is made to the OSQP solver.

        _quad_obj_exprs: list of quadratic expressions in the objective
        _nonquad_obj_exprs: list of non-quadratic expressions in the objective
        _approx_obj_exprs: list of approximation of the non-quadratic
            expressions in the objective

        _nonlin_cnt_exprs: list of non-linear constraint expressions
        _penalty_exprs: list of penalty term expressions (approximations of the
            non-linear constraint expressions in _nonlin_cnt_exprs)

        _osqp_quad_objs: list of OSQPQuadraticObjs that keep track of the quadratic objectives
            that will be passed to the QP
        _osqp_lin_objs: list of OSQPLinearObjectives that keep track of the linear objectives
            that will be passed to the QP
        _osqp_lin_cnt_exprs: list of OSQPLinearConstraints that keep track of the linear constraints
            that will be passed to the QP

        _osqp_penalty_cnts: list of OSQP constraints that are generated when
            adding the hinge and absolute value terms from the penalty terms.

        _bexpr_to_osqp_expr: dictionary that caches quadratic bound expressions
            with their corresponding Gurobi expression
        """
        self._vars = set()
        self._osqp_vars = set()
        if callback is not None:
            self._callback = callback
        else:

            def do_nothing():
                pass

            self._callback = do_nothing

        self._quad_obj_exprs = []
        self._nonquad_obj_exprs = []
        self._approx_obj_exprs = []

        self._nonlin_cnt_exprs = []

        # These are lists of OSQPQuadraticObj's, OSQPLinearObj's and OSQPLinearConstraints
        # that will directly be used to construct the P, q and A matrices that define
        # the final QP to be solved.
        self._osqp_quad_objs = []
        self._osqp_lin_objs = []
        self._osqp_lin_cnt_exprs = []

        # list of constraints that will hold the hinge constraints
        # for each non-linear constraint, is a pair of constraints
        # for an eq constraint
        self.hinge_created = False

        self._penalty_exprs = []
        self._osqp_penalty_cnts = []
        self._osqp_penalty_exprs = []

        # group-id (str) -> cnt-set (set of constraints)
        self._cnt_groups = defaultdict(set)
        self._cnt_groups_overlap = defaultdict(set)
        self._penalty_groups = []
        self.nonconverged_groups = []
        self.gid2ind = {}

    def add_obj_expr(self, bound_expr):
        """
        Adds a bound expression (bound_expr) to the objective. If the objective
        is quadratic, is it added to _quad_obj_exprs. Otherwise, it is added
        to self._nonquad_obj_exprs.

        bound_expr's var is added to self._vars so that a trust region can be
        added to var.
        """
        expr = bound_expr.expr
        if isinstance(expr, sco_osqp_expr.AffExpr) or isinstance(
            expr, sco_osqp_expr.QuadExpr
        ):
            self._quad_obj_exprs.append(bound_expr)
        else:
            self._nonquad_obj_exprs.append(bound_expr)
        self.add_var(bound_expr.var)

    def add_var(self, var):
        self._vars.add(var)

    def add_osqp_var(self, osqp_var):
        self._osqp_vars.add(osqp_var)

    def add_cnt_expr(self, bound_expr, group_ids=None):
        """
        Adds a bound expression (bound_expr) to the problem's constraints.
        If the constraint is linear, it is added directly to the model.
        Otherwise, the constraint is added by appending bound_expr to
        self._nonlin_cnt_exprs.

        bound_expr's var is added to self._vars so that a trust region can be
        added to var.
        """
        comp_expr = bound_expr.expr
        expr = comp_expr.expr
        var = bound_expr.var
        assert isinstance(comp_expr, sco_osqp_expr.CompExpr)
        if isinstance(expr, sco_osqp_expr.AffExpr):
            if isinstance(comp_expr, sco_osqp_expr.EqExpr):
                self._add_osqp_cnt_from_aff_expr(expr, var, "eq", comp_expr.val)
            elif isinstance(comp_expr, sco_osqp_expr.LEqExpr):
                self._add_osqp_cnt_from_aff_expr(expr, var, "leq", comp_expr.val)
        else:
            self._nonlin_cnt_exprs.append(bound_expr)
            self._reset_hinge_cnts()

            if group_ids is None:
                group_ids = ["all"]
            for gid in group_ids:
                self._cnt_groups[gid].add(bound_expr)
                for other in group_ids:
                    if other == gid:
                        continue
                    self._cnt_groups_overlap[gid].add(other)

        self.add_var(var)

    def optimize(self, 
                 add_convexified_terms=False, 
                 osqp_eps_abs=1e-06, 
                 osqp_eps_rel=1e-09, 
                 osqp_max_iter=osqp_utils.DEFAULT_MAX_ITER, 
                 rho: float = 1e-01,
                 adaptive_rho: bool = True,
                 verbose=False):
        """
        Calls the OSQP optimizer on the current QP approximation with a given
        penalty coefficient. Note that add_convexified_terms is a convenience
        boolean useful to toggle whether or not self._osqp_penalty_exprs and
        self._osqp_penalty_cnts are included in the optimization problem
        """
        if not add_convexified_terms:
            solve_res, var_to_index_dict = osqp_utils.optimize(
                self._osqp_vars,
                self._vars,
                self._osqp_quad_objs,
                self._osqp_lin_objs,
                self._osqp_lin_cnt_exprs,
                osqp_eps_abs,
                osqp_eps_rel,
                osqp_max_iter,
                rho=rho,
                adaptive_rho=adaptive_rho,
                verbose=verbose,
            )
        else:
            cnt_exprs = self._osqp_lin_cnt_exprs[:]
            for penalty_cnt_list in self._osqp_penalty_cnts:
                cnt_exprs.extend(penalty_cnt_list)

            solve_res, var_to_index_dict = osqp_utils.optimize(
                self._osqp_vars,
                self._vars,
                self._osqp_quad_objs,
                self._osqp_lin_objs + self._osqp_penalty_exprs,
                cnt_exprs,
                osqp_eps_abs,
                osqp_eps_rel,
                osqp_max_iter,
                rho=rho,
                adaptive_rho=adaptive_rho,
                verbose=verbose,
            )

        # If the solve failed, just return False
        if solve_res.info.status_val not in [1, 2]:
            return False

        # If the solve succeeded, update all the variables with these new values, then
        # run he callback before returning true
        osqp_utils.update_osqp_vars(var_to_index_dict, solve_res.x)
        self._update_vars()
        self._callback() # TODO: Modify to get the visualizer in a better spot.
        return True

    def _reset_hinge_cnts(self):
        # reset the hinge_cnts
        self.hinge_created = False

    def _add_osqp_objs_and_cnts_from_expr(self, bound_expr):
        """
        Uses AffExpr, QuadExpr, HingeExpr and AbsExpr to extract
        OSQP solver compatible data structures. Depending on the expression type,
        appends elements to self._osqp_quad_objs, or self._osqp_lin_objs.
        """
        expr = bound_expr.expr
        var = bound_expr.var

        if isinstance(expr, sco_osqp_expr.AffExpr):
            self._add_to_lin_objs_and_cnts_from_aff_expr(expr, var)
        elif isinstance(expr, sco_osqp_expr.QuadExpr):
            self._add_to_quad_and_lin_objs_from_quad_expr(expr, var)
        elif isinstance(expr, sco_osqp_expr.HingeExpr):
            self._add_to_lin_objs_and_cnts_from_hinge_expr(expr, var)
        elif isinstance(expr, sco_osqp_expr.AbsExpr):
            self._add_to_lin_objs_and_cnts_from_abs_expr(expr, var)
        elif isinstance(expr, sco_osqp_expr.CompExpr):
            raise Exception(
                "Comparison Expressions cannot be converted to \
                OSQP problem objectives; use _add_osqp_cnt_from_aff_expr \
                instead"
            )
        else:
            raise Exception(
                "This type of Expression cannot be converted to\
                an OSQP objective."
            )

    def _add_to_lin_objs_and_cnts_from_aff_expr(self, aff_expr, var):
        osqp_vars = var.get_osqp_vars()
        A_mat = aff_expr.A

        for i in range(A_mat.shape[0]):
            (inds,) = np.nonzero(A_mat[i, :])
            for coeff, osqp_var in zip(
                A_mat[i, inds].tolist(), osqp_vars[inds, 0].tolist()
            ):
                self._osqp_penalty_exprs.append(OSQPLinearObj(osqp_var, coeff))

    def _add_to_lin_objs_and_cnts_from_hinge_expr(self, hinge_expr, var):
        aff_expr = hinge_expr.expr
        assert isinstance(aff_expr, sco_osqp_expr.AffExpr)
        osqp_vars = var.get_osqp_vars()
        A_mat = aff_expr.A
        b_vec = aff_expr.b

        hinge = self.create_pos_osqp_var_arr((A_mat.shape[0], 1))
        for _, hinge_var in np.ndenumerate(hinge):
            self._osqp_penalty_exprs.append(OSQPLinearObj(hinge_var, 1.0))

        cnts_list = []
        for i in range(A_mat.shape[0]):
            (inds,) = np.nonzero(A_mat[i, :])
            # since this is a hinge constraint, the lower bound is -inf
            curr_lb = -np.inf
            # the constraint expr is:
            # curr_lb <= osqp_vars[inds, 0] * A_mat[i, inds] <= hinge[i] - b_vec[i]
            # which can be reformulated as:
            # curr_lb <= osqp_vars[inds, 0] * A_mat[i, inds] - hinge[i] <= -b_vec[i]
            # since curr_lb is -inf
            curr_ub = -b_vec[i]
            cnt_vars = np.concatenate((osqp_vars[inds, 0], hinge[i]))
            cnt_coeffs = np.concatenate((A_mat[i, inds], np.array([-1.0])))
            curr_cnt_expr = OSQPLinearConstraint(cnt_vars, cnt_coeffs, curr_lb, curr_ub)
            cnts_list.append(curr_cnt_expr)

        self._osqp_penalty_cnts.append(cnts_list)

    def _add_to_lin_objs_and_cnts_from_abs_expr(self, abs_expr, var):
        aff_expr = abs_expr.expr
        assert isinstance(aff_expr, sco_osqp_expr.AffExpr)
        A_mat = aff_expr.A
        b_vec = aff_expr.b
        pos = self.create_pos_osqp_var_arr((A_mat.shape[0], 1))
        neg = self.create_pos_osqp_var_arr((A_mat.shape[0], 1))

        # First, add all the objective terms
        for pos_var in pos.flat:
            self._osqp_penalty_exprs.append(OSQPLinearObj(pos_var, 1.0))
        for neg_var in neg.flat:
            self._osqp_penalty_exprs.append(OSQPLinearObj(neg_var, 1.0))

        # Next, add the constraints
        osqp_vars = var.get_osqp_vars()
        A_mat = aff_expr.A
        b_vec = aff_expr.b

        cnts_list = []
        for i in range(A_mat.shape[0]):
            (inds,) = np.nonzero(A_mat[i, :])
            # the constraint expr is:
            # pos_var[i] - neg_var[i] <= osqp_vars[inds, 0] * A_mat[i, inds] + b_vec[i] <= pos_var[i] - neg_var[i]
            # which can be reformulated as:
            # -b_vec[i] <= osqp_vars[inds, 0] * A_mat[i, inds] - pos_var[i] + neg_var[i] <= -b_vec[i]
            curr_ub = -b_vec[i]
            curr_lb = -b_vec[i]
            cnt_vars = np.concatenate((osqp_vars[inds, 0], pos[i], neg[i]))
            cnt_coeffs = np.concatenate(
                (A_mat[i, inds], np.array([-1.0]), np.array([1.0]))
            )
            curr_cnt_expr = OSQPLinearConstraint(cnt_vars, cnt_coeffs, curr_lb, curr_ub)
            cnts_list.append(curr_cnt_expr)

        self._osqp_penalty_cnts.append(cnts_list)

    def _add_osqp_cnt_from_aff_expr(self, aff_expr, var, cnt_type, cnt_val):
        """
        Uses aff_expr to create OSQPLinearConstraints of cnt_type that are then
        appended to self._osqp_lin_cnt_exprs
        """
        osqp_vars = var.get_osqp_vars()
        A_mat = aff_expr.A
        b_vec = aff_expr.b
        for i in range(A_mat.shape[0]):
            (inds,) = np.nonzero(A_mat[i, :])
            # If the constraint to be added is an equality constraint,
            # compute the upper and lower bounds
            if cnt_type == "eq":
                # the upper and lower bounds must be equal, and they must be
                # whatever the cnt_val was minus the constant term
                curr_lb = cnt_val[i] - b_vec[i]
                curr_ub = cnt_val[i] - b_vec[i]
            elif cnt_type == "leq":
                # only the upper bound needs to be set; the lower bound is negative
                # infinity
                curr_lb = -np.inf
                curr_ub = cnt_val[i] - b_vec[i]
            else:
                raise NotImplementedError

            curr_cnt_expr = OSQPLinearConstraint(
                osqp_vars[inds, 0], A_mat[i, inds], curr_lb, curr_ub
            )

            self._osqp_lin_cnt_exprs.append(curr_cnt_expr)

    def _add_to_quad_and_lin_objs_from_quad_expr(self, quad_expr, var):
        x = var.get_osqp_vars()
        Q = quad_expr.Q
        rows, cols = x.shape
        assert cols == 1
        inds = np.nonzero(Q)
        coeffs = Q[inds]  # No need to multiply by 2 because OSQP expects 0.5*x.T*Q*x
        v1 = x[inds[0], 0]
        v2 = x[inds[1], 0]
        # Create the new QuadraticObj term and append it to the problem's running list of
        # such terms
        self._osqp_quad_objs.append(OSQPQuadraticObj(v1, v2, coeffs))
        inds = np.nonzero(quad_expr.A)
        lin_coeffs = quad_expr.A[inds]
        # Because quad_expr.A is of shape (1,2), inds[1] corresponds to the nonzero
        # vars
        lin_vars = x[inds[1], 0]
        assert lin_coeffs.shape == lin_vars.shape
        for lin_var, lin_coeff in zip(lin_vars.tolist(), lin_coeffs.tolist()):
            self._osqp_lin_objs.append(OSQPLinearObj(lin_var, lin_coeff))

    def find_closest_feasible_point(self):
        """
        Finds the closest point (l2 norm) to the initialization that satisfies
        the linear constraints.
        """
        for var in self._vars:
            osqp_vars = var.get_osqp_vars()
            val = var.get_value()
            if val is not None:
                assert osqp_vars.shape == val.shape
                inds = np.where(~np.isnan(val))
                val = val[inds]
                nonnan_osqp_vars = osqp_vars[inds]
                val_arr = val.flatten()
                for i, nonnan_osqp_var in enumerate(
                    nonnan_osqp_vars.flatten().tolist()
                ):
                    # Create the correct quad and lin objectives!
                    self._osqp_lin_objs.append(
                        OSQPLinearObj(nonnan_osqp_var, -2.0 * val_arr[i])
                    )

                    self._osqp_quad_objs.append(
                        OSQPQuadraticObj(
                            np.array([nonnan_osqp_var]),
                            np.array([nonnan_osqp_var]),
                            np.array([2.0]),
                        )
                    )

        return self.optimize()

    def update_obj(self, penalty_coeff=0.0):
        self._reset_osqp_objs()
        self._lazy_spawn_osqp_cnts()

        for bound_expr in self._quad_obj_exprs + self._approx_obj_exprs:
            self._add_osqp_objs_and_cnts_from_expr(bound_expr)

        for i, bound_expr in enumerate(self._penalty_exprs):
            self._update_nonlin_cnt_and_add_to_qp(bound_expr, i)

        for lin_obj in self._osqp_penalty_exprs:
            lin_obj.coeff = lin_obj.coeff * penalty_coeff
            self._osqp_lin_objs.append(lin_obj)

    def _reset_osqp_objs(self):
        """Resets the quadratic and linear objectives in preparation for the
        definition of a new OSQP problem"""
        self._osqp_quad_objs = []
        self._osqp_lin_objs = []

    def _lazy_spawn_osqp_cnts(self):
        if not self.hinge_created:
            self._osqp_penalty_cnts = []
            self._osqp_penalty_exprs = []
            self._osqp_nz = []
            for bound_expr in self._penalty_exprs:
                self._osqp_nz.append(np.nonzero(bound_expr.expr.expr.A))
                # The below line populates self._osqp_penalty_exprs and
                # self._osqp_penalty_cnts
                self._add_osqp_objs_and_cnts_from_expr(bound_expr)
            self.hinge_created = True

    def create_pos_osqp_var_arr(self, shape):
        """
        Returns a numpy array of new, unused positive OSQP variables.
        """
        osqp_var_arr = np.empty(shape, dtype=object)
        for x in np.nditer(osqp_var_arr, op_flags=["readwrite"], flags=["refs_ok"]):
            # Create a new variable that's bound between 0 and np.inf, and name it so that
            # it so it starts with a z and will get sorted to the end
            new_pos_var = OSQPVar("z+_pos_osqp_var", 0.0, np.inf, 0.0)
            # Add it to the set keeping track of all OSQP vars
            self._osqp_vars.add(new_pos_var)
            x[...] = new_pos_var
        return osqp_var_arr

    # @profile
    def _update_nonlin_cnt_and_add_to_qp(self, bexpr, ind):
        """Updates all non-linear constraints and adds to new (approximated) linear
        constraints to _osqp_lin_cnt_exprs"""

        expr, var = bexpr.expr, bexpr.var
        if isinstance(expr, sco_osqp_expr.HingeExpr) or isinstance(
            expr, sco_osqp_expr.AbsExpr
        ):
            aff_expr = expr.expr
            assert isinstance(aff_expr, sco_osqp_expr.AffExpr)
            A, b = aff_expr.A, aff_expr.b
            cnts = self._osqp_penalty_cnts[ind]
            old_nz = self._osqp_nz[ind]
            osqp_vars = var.get_osqp_vars()
            nz = np.nonzero(A)

            for i in range(A.shape[0]):
                # if lb == ub, then it is an equality constraint, so
                # we need to set both lb and ub to -b[i, 0]
                if cnts[i].lb == cnts[i].ub:
                    cnts[i].lb = -b[i, 0]
                    cnts[i].ub = -b[i, 0]
                # otherwise, it is a LEQ constraint and so
                # we only need to update the ub
                else:
                    cnts[i].ub = -b[i, 0]

            for idx in range(old_nz[0].shape[0]):
                i, j = old_nz[0][idx], old_nz[1][idx]
                # if osqp_vars[j, 0] is in cnts[i].osqp_vars, then modify
                # cnts[i].osqp_vars to set the associated coefficient
                # to 0
                if osqp_vars[j, 0] in cnts[i].osqp_vars:
                    rep_i = np.where(cnts[i].osqp_vars == osqp_vars[j, 0])[0][0]
                    cnts[i].coeffs[rep_i] = 0.0

            for idx in range(nz[0].shape[0]):
                i, j = nz[0][idx], nz[1][idx]
                # if osqp_vars[j, 0] is in cnts[i].osqp_vars, then modify
                # cnts[i].osqp_vars to set the associated coefficient
                # to A[i, j]
                if osqp_vars[j, 0] in cnts[i].osqp_vars:
                    rep_i = np.where(cnts[i].osqp_vars == osqp_vars[j, 0])[0][0]
                    cnts[i].coeffs[rep_i] = A[i, j]

            self._osqp_nz[ind] = nz

            for cnt in cnts:
                self._osqp_lin_cnt_exprs.append(cnt)

        else:
            raise NotImplementedError

    def add_trust_region(self, trust_region_size):
        """
        Adds the trust region for every variable
        """
        for var in self._vars:
            var.add_trust_region(trust_region_size)

    # @profile
    def convexify(self):
        """
        Convexifies the optimization problem by computing a QP approximation
        A quadratic approximation of the non-quadratic objective terms
        (self._nonquad_obj_exprs) is saved in self._approx_obj_exprs.
        The penalty approximation of the non-linear constraints
        (self._nonlin_cnt_exprs) is saved in self._penalty_exprs
        """
        # if len(self._nonquad_obj_exprs) > 0:
        #     import ipdb; ipdb.set_trace()
        self._approx_obj_exprs = [
            bexpr.convexify(degree=2) for bexpr in self._nonquad_obj_exprs
        ]
        self._penalty_exprs = [
            bexpr.convexify(degree=1) for bexpr in self._nonlin_cnt_exprs
        ]
        self._penalty_groups = []
        gids = sorted(self._cnt_groups.keys())
        self.gid2ind = {}
        for i, gid in enumerate(gids):
            self.gid2ind[gid] = i
            cur_bexprs = [bexpr.convexify(degree=1) for bexpr in self._cnt_groups[gid]]
            self._penalty_groups.append(cur_bexprs)

    # #@profile
    def get_value(self, penalty_coeff, vectorize=False):
        """
        Returns the current value of the penalty objective.
        The penalty objective is computed by summing up all the values of the
        quadratic objective expressions (self._quad_obj_exprs), the
        non-quadratic objective expressions and the penalty coeff multiplied
        by the constraint violations (computed using _nonlin_cnt_exprs)

        if vectorize=True, then this returns a vector of constraint
        violations -- 1 per group id.
        """
        if vectorize:
            gids = sorted(self._cnt_groups.keys())
            value = np.zeros(len(gids))
            for i, gid in enumerate(gids):
                value[i] = np.sum(
                    np.sum(
                        [
                            np.sum(self._compute_cnt_violation(bexpr))
                            for bexpr in self._cnt_groups[gid]
                        ]
                    )
                )
            return value
        else:
            value = 0.0
            for bound_expr in self._quad_obj_exprs + self._nonquad_obj_exprs:
                value += np.sum(np.sum(bound_expr.eval()))
            for bound_expr in self._nonlin_cnt_exprs:
                cnt_vio = self._compute_cnt_violation(bound_expr)
                value += penalty_coeff * np.sum(cnt_vio)

        return value

    # @profile
    def _compute_cnt_violation(self, bexpr):
        comp_expr = bexpr.expr
        var_val = bexpr.var.get_value()
        if isinstance(comp_expr, sco_osqp_expr.EqExpr):
            return np.absolute(comp_expr.expr.eval(var_val) - comp_expr.val)
        elif isinstance(comp_expr, sco_osqp_expr.LEqExpr):
            v = comp_expr.expr.eval(var_val) - comp_expr.val
            zeros = np.zeros(v.shape)
            return np.maximum(v, zeros)

    def get_max_cnt_violation(self):
        """
        Returns the the maximum amount a non-linear constraint is violated.
        Linear constraints are assumed to be satisfied because they are added
        directly to the model and QP solvers can deal with them.
        """
        max_vio = 0.0
        for bound_expr in self._nonlin_cnt_exprs:
            cnt_vio = self._compute_cnt_violation(bound_expr)
            cnt_max_vio = np.amax(cnt_vio)
            max_vio = np.maximum(max_vio, cnt_max_vio)
        return max_vio

    def get_approx_value(self, penalty_coeff, vectorize=False):
        """
        Returns the current value of the penalty QP approximation by summing
        up the expression values for the quadratic objective terms
        (_quad_obj_exprs), the quadratic approximation of the non-quadratic
        terms (_approx_obj_exprs) and the penalty terms (_penalty_exprs).
        Note that this approximate value is computed with respect to when the
        last convexification was performed.

        if vectorize=True, then this returns a vector of constraint
        violations -- 1 per group id.
        """
        if vectorize:
            value = np.zeros(len(self._penalty_groups))
            for i, bexprs in enumerate(self._penalty_groups):
                x = np.array([np.sum(bexpr.eval()) for bexpr in bexprs])
                value[i] = np.sum(x.flatten())
            return value

        value = 0.0
        for bound_expr in self._quad_obj_exprs + self._approx_obj_exprs:
            value += np.sum(np.sum(bound_expr.eval()))
        for bound_expr in self._penalty_exprs:
            value += penalty_coeff * np.sum(bound_expr.eval())

        return value

    def _update_vars(self):
        for var in self._vars:
            var.update()

    def batch_add_lin_cnts(self, list_of_lin_cnts):
        self._osqp_lin_cnt_exprs.extend(list_of_lin_cnts)

    def save(self):
        """
        Saves the problem's current state by saving the values of all the
        variables.
        """
        for var in self._vars:
            var.save()

    def restore(self):
        """
        Restores the problem's state to the problem's saved state
        """
        for var in self._vars:
            var.restore()
