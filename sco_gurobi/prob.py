# fmt: off
from collections import defaultdict

import gurobipy as grb
import numpy as np

from sco.expr import (AbsExpr, AffExpr, CompExpr, EqExpr, HingeExpr,
                             LEqExpr, QuadExpr)

# fmt: on

GRB = grb.GRB

class Prob(object):
    """
    Sequential convex programming problem with a scalar objective. A solution is
    found using the l1 penalty method.
    """

    def __init__(self, grb_model, callback=None):
        """
        _model: Gurobi model associated with this problem
        _vars: variables in this problem

        _quad_obj_exprs: list of quadratic expressions in the objective
        _nonquad_obj_exprs: list of non-quadratic expressions in the objective
        _approx_obj_exprs: list of approximation of the non-quadratic
            expressions in the objective

        _nonlin_cnt_exprs: list of non-linear constraint expressions
        _penalty_exprs: list of penalty term expressions (approximations of the
            non-linear constraint expressions in _nonlin_cnt_exprs)

        _grb_penalty_cnts: list of Gurobi constraints that are generated when
            adding the hinge and absolute value terms from the penalty terms.
        _pgm: Positive Gurobi variable manager provides a lazy way of generating
            positive Gurobi variables so that there are less model updates.

        _bexpr_to_grb_expr: dictionary that caches quadratic bound expressions
            with their corresponding Gurobi expression
        """
        self._model = grb_model
        self._model.params.OutputFlag = 0  # silences Gurobi output
        self._vars = set()
        if callback is not None:
            self._callback = callback
        else:

            def do_nothing():
                pass

            self._callback = do_nothing

        self._quad_obj_exprs = []
        self._nonquad_obj_exprs = []
        self._approx_obj_exprs = []

        # linear constraints are added directly to the model so there's no
        # need for a _lin_cnt_exprs variable
        self._nonlin_cnt_exprs = []

        # list of constraints that will hold the hinge constraints
        # for each non-linear constraint, is a pair of constraints
        # for an eq constraint
        self.hinge_created = False

        self._penalty_exprs = []
        self._grb_penalty_cnts = []  # hinge and abs value constraints
        self._pgm = PosGRBVarManager(self._model)

        self._bexpr_to_grb_expr = {}

        ## group-id (str) -> cnt-set (set of constraints)
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
        if isinstance(expr, AffExpr) or isinstance(expr, QuadExpr):
            self._quad_obj_exprs.append(bound_expr)
        else:
            self._nonquad_obj_exprs.append(bound_expr)
        self.add_var(bound_expr.var)

    def add_var(self, var):
        self._vars.add(var)

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
        if isinstance(expr, AffExpr):
            # adding constraint directly into model
            grb_expr, grb_cnt = self._aff_expr_to_grb_expr(expr, var)
            if isinstance(comp_expr, EqExpr):
                self._add_np_array_grb_cnt(grb_expr, GRB.EQUAL, comp_expr.val)
            elif isinstance(comp_expr, LEqExpr):
                self._add_np_array_grb_cnt(grb_expr, GRB.LESS_EQUAL, comp_expr.val)
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

    def _reset_hinge_cnts(self):
        ## reset the hinge_cnts
        self.hinge_created = False

    # @profile
    def _add_np_array_grb_cnt(self, grb_exprs, sense, val):
        """
        Adds a numpy array of Gurobi constraints to the model and returns
        the constraints.
        """
        cnts = []
        for index, grb_expr in np.ndenumerate(grb_exprs):
            cnts.append(self._model.addConstr(grb_expr, sense, val[index]))
        return cnts

    # @profile
    def _expr_to_grb_expr(self, bound_expr):
        """
        Translates AffExpr, QuadExpr, HingeExpr and AbsExpr to Gurobi
        expressions and returns the corresponding Gurobi expressions and
        constraints. If there are no Gurobi constraints, an empty list is
        returned. Otherwise, this method raises an exception.
        """
        expr = bound_expr.expr
        var = bound_expr.var

        if isinstance(expr, AffExpr):
            return self._aff_expr_to_grb_expr(expr, var)
        elif isinstance(expr, QuadExpr):
            if bound_expr in self._bexpr_to_grb_expr:
                return self._bexpr_to_grb_expr[bound_expr], []
            else:
                grb_expr, cnts = self._quad_expr_to_grb_expr(expr, var)
                self._bexpr_to_grb_expr[bound_expr] = grb_expr
                return grb_expr, cnts
        elif isinstance(expr, HingeExpr):
            return self._hinge_expr_to_grb_expr(expr, var)
        elif isinstance(expr, AbsExpr):
            return self._abs_expr_to_grb_expr(expr, var)
        elif isinstance(expr, CompExpr):
            raise Exception(
                "Comparison Expressions cannot be converted to \
                a Gurobi expression. Use add_cnt_expr instead"
            )
        else:
            raise Exception(
                "This type of Expression cannot be converted to\
                a Gurobi expression."
            )

    # @profile
    def _aff_expr_to_grb_expr(self, aff_expr, var):
        grb_var = var.get_grb_vars()
        grb_exprs = []
        A = aff_expr.A
        b = aff_expr.b
        for i in range(A.shape[0]):
            grb_expr = grb.LinExpr()
            (inds,) = np.nonzero(A[i, :])
            grb_expr += b[i]
            grb_expr.addTerms(A[i, inds].tolist(), grb_var[inds, 0].tolist())
            grb_exprs.append([grb_expr])
        return np.array(grb_exprs), []

    # #@profile
    def _quad_expr_to_grb_expr(self, quad_expr, var):
        x = var.get_grb_vars()
        grb_expr = grb.QuadExpr()
        Q = quad_expr.Q
        rows, cols = x.shape
        assert cols == 1
        inds = np.nonzero(Q)
        coeffs = 0.5 * Q[inds]
        v1 = x[inds[0], 0]
        v2 = x[inds[1], 0]
        grb_expr.addTerms(coeffs.tolist(), v1.tolist(), v2.tolist())
        inds = np.nonzero(quad_expr.A)
        coeffs = quad_expr.A[inds]
        v1 = x[inds[1], 0]
        grb_expr.addTerms(coeffs.tolist(), v1.tolist())
        grb_expr = grb_expr + quad_expr.b

        return np.array([[grb_expr]]), []

    # @profile
    def _hinge_expr_to_grb_expr(self, hinge_expr, var):
        aff_expr = hinge_expr.expr
        assert isinstance(aff_expr, AffExpr)
        grb_expr, _ = self._aff_expr_to_grb_expr(aff_expr, var)
        grb_hinge = self._pgm.get_array(grb_expr.shape)
        cnts = self._add_np_array_grb_cnt(grb_expr, GRB.LESS_EQUAL, grb_hinge)

        return grb_hinge, cnts

    def _abs_expr_to_grb_expr(self, abs_expr, var):
        aff_expr = abs_expr.expr
        assert isinstance(aff_expr, AffExpr)
        grb_expr, _ = self._aff_expr_to_grb_expr(aff_expr, var)
        pos = self._pgm.get_array(grb_expr.shape)
        neg = self._pgm.get_array(grb_expr.shape)
        cnts = self._add_np_array_grb_cnt(grb_expr, GRB.EQUAL, pos - neg)

        return pos + neg, cnts

    def find_closest_feasible_point(self):
        """
        Finds the closest point (l2 norm) to the initialization that satisfies
        the linear constraints.
        """
        self._del_old_grb_cnts()
        self._model.update()

        obj = grb.QuadExpr()
        for var in self._vars:
            g_var = var.get_grb_vars()
            val = var.get_value()
            if val is not None:
                assert g_var.shape == val.shape
                inds = np.where(~np.isnan(val))
                val = val[inds]
                g_var = g_var[inds]
                obj += np.sum((g_var - val).T.dot(g_var - val))

                # for i in range(g_var.shape[0]):
                #     if g_var[i].var_name == "(can1-pose-(0, 2))":
                #         import pdb

                #         pdb.set_trace()

                # for i in np.ndindex(g_var.shape):
                #    if not np.isnan(val[i]):
                #            obj += g_var[i]*g_var[i] - 2*val[i]*g_var[i] + val[i]*val[i]

            # grb_exprs = []
            # for bound_expr in self._quad_obj_exprs:
            #     grb_expr, grb_cnts = self._expr_to_grb_expr(bound_expr)
            #     self._grb_penalty_cnts.extend(grb_cnts)
            #     grb_exprs.extend(grb_expr.flatten().tolist())

            # obj += grb.quicksum(grb_exprs)

        self._model.setObjective(obj)

        self.optimize()

        return self._model.status == 2

    def optimize(self):
        """
        Calls the Gurobi optimizer on the current QP approximation with a given
        penalty coefficient.

        Temporary Gurobi constraints and variables from the previous optimize
        call are deleted.

        The Gurobi objective is computed by translating all the expressions
        in the quadratically approximated objective (self._quad_obj_expr) and
        in the penalty approximation of the constraints (self._penalty_exprs)
        to Gurobi expressions, and summing them. The temporary constraints and
        variables created from the translation process are saved so that they
        can be deleted later.

        The Gurobi constraints are the linear constraints which have already
        been added to the model when constraints were added to this problem.
        """
        self._model.optimize()

        try:
            self._update_vars()
        except Exception as e:
            print(e)
            print(("Model status:", self._model.status))
        # import ipdb; ipdb.set_trace()
        self._callback()

    def print_grb_vals(self):
        """convenience function for debugging"""
        for var in self._model.getVars():
            print(var)

    # @profile
    def update_obj(self, penalty_coeff=0.0):
        self._lazy_spawn_grb_cnts()
        grb_exprs = []
        
        for bound_expr in self._quad_obj_exprs + self._approx_obj_exprs:
            grb_expr, grb_cnts = self._expr_to_grb_expr(bound_expr)
            # self._grb_penalty_cnts.extend(grb_cnts)
            grb_exprs.extend(grb_expr.flatten().tolist())

        for i, bound_expr in enumerate(self._penalty_exprs):
            grb_expr = self._update_nonlin_cnt(bound_expr, i).flatten()
            grb_exprs.extend(grb_expr * penalty_coeff)

        obj = grb.quicksum(grb_exprs)

        self._model.setObjective(obj)
        self._model.update()

    def _del_old_grb_cnts(self):
        for cnt in self._grb_penalty_cnts:
            self._model.remove(cnt)
        self.hinge_created = False

    def _lazy_spawn_grb_cnts(self):
        if self.hinge_created:
            return
        self._del_old_grb_cnts()
        self._grb_penalty_cnts = []
        self._grb_penalty_exprs = []
        self._grb_nz = []
        for bound_expr in self._penalty_exprs:
            grb_expr, grb_cnts = self._expr_to_grb_expr(bound_expr)
            self._grb_penalty_cnts.append(grb_cnts)
            self._grb_penalty_exprs.append(grb_expr)
            self._grb_nz.append(np.nonzero(bound_expr.expr.expr.A))
        self._model.update()
        self.hinge_created = True

    # @profile
    def _update_nonlin_cnt(self, bexpr, ind):
        expr, var = bexpr.expr, bexpr.var
        if isinstance(expr, HingeExpr) or isinstance(expr, AbsExpr):
            aff_expr = expr.expr
            assert isinstance(aff_expr, AffExpr)
            A, b = aff_expr.A, aff_expr.b
            cnts = self._grb_penalty_cnts[ind]
            grb_expr = self._grb_penalty_exprs[ind]
            old_nz = self._grb_nz[ind]
            grb_vars = var.get_grb_vars()
            nz = np.nonzero(A)

            # This updates the constraint's RHS
            for i in range(A.shape[0]):
                ## add negative b to rhs because it
                ## changes sides of the ineq/eq
                cnts[i].setAttr("rhs", -b[i, 0])

            # This sets all the old nz terms' coeffs to 0
            for idx in range(old_nz[0].shape[0]):
                i, j = old_nz[0][idx], old_nz[1][idx]
                self._model.chgCoeff(cnts[i], grb_vars[j, 0], 0)

            # The nonzero terms are then updated with their new coeffs
            ## then set the non-zero values
            for idx in range(nz[0].shape[0]):
                i, j = nz[0][idx], nz[1][idx]
                self._model.chgCoeff(cnts[i], grb_vars[j, 0], A[i, j])

            self._grb_nz[ind] = nz

            return grb_expr
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
        # if len(self._nonlin_cnt_exprs) > 0:
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
        if isinstance(comp_expr, EqExpr):
            return np.absolute(comp_expr.expr.eval(var_val) - comp_expr.val)
        elif isinstance(comp_expr, LEqExpr):
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
        """
        Updates the variables values
        """
        unique_grb_vars = set()
        for var in self._vars:
            var.update()
            for gv in var._grb_vars.flatten().tolist():
                unique_grb_vars.add(gv)

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


class PosGRBVarManager(object):
    """
    Manages positive Gurobi variables. The purpose of the manager is to create
    many Gurobi variables at once to decrease the number of Gurobi model update
    because model updates take a long time.
    """

    INIT_NUM = 1000
    INC_NUM = 1000

    def __init__(self, model, init_num=INC_NUM, inc_num=INC_NUM):
        self._index = 0
        self._model = model
        self._grb_vars = []
        self._add_grb_vars(init_num)
        self._inc_num = inc_num

    def _add_grb_vars(self, num=None):
        """
        Creates a batch of positive Gurobi variables so that the model is
        updated less often.
        """
        if num is None:
            num = self._inc_num
        new_grb_vars = [self._model.addVar(lb=0.0, ub=GRB.INFINITY) for i in range(num)]
        self._grb_vars.extend(new_grb_vars)
        self._model.update()

    def __next__(self):
        """
        Returns one positive Gurobi variable.
        """
        if self._index == len(self._grb_vars) - 1:
            self._add_grb_vars()
        self._index += 1
        return self._grb_vars[self._index - 1]

    # @profile
    def get_array(self, shape):
        """
        Returns a numpy array of unused positive Gurobi variables.
        """
        a = np.empty(shape, dtype=object)
        for x in np.nditer(a, op_flags=["readwrite"], flags=["refs_ok"]):
            x[...] = next(self)
        return a

    def reset(self):
        self._index = 0
