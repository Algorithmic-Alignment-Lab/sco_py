import time

import numpy as np


class Solver(object):
    """
    SCO Solver
    """

    def __init__(self):
        """
        values taken from Pieter Abbeel's CS287 hw3 q2 penalty_sqp.m file
        """
        self.improve_ratio_threshold = 0.25
        self.min_trust_region_size = 1e-4
        self.min_approx_improve = 1e-4
        self.max_iter = 50
        self.trust_shrink_ratio = 0.1
        self.trust_expand_ratio = 1.5
        self.cnt_tolerance = 1e-4
        self.max_merit_coeff_increases = 1
        self.merit_coeff_increase_ratio = 1e1
        self.initial_trust_region_size = 1
        self.initial_penalty_coeff = 1e3

    def solve(self, prob, method=None, tol=None, verbose=False):
        """
        Returns whether solve succeeded.

        Given a sco (sequential convex optimization) problem instance, solve
        using specified method to find a solution. If the specified method
        doesn't exist, an exception is thrown.
        """
        if tol is not None:
            self.min_trust_region_size = tol
            self.min_approx_improve = tol
            self.cnt_tolerance = tol

        if method == "penalty_sqp":
            return self._penalty_sqp(prob, verbose=verbose)
        else:
            raise Exception("This method is not supported.")

    # @profile
    def _penalty_sqp(self, prob, verbose=False):
        """
        Return true is the penalty sqp method succeeds.
        Uses Penalty Sequential Quadratic Programming to solve the problem
        instance.
        """
        start = time.time()
        trust_region_size = self.initial_trust_region_size
        penalty_coeff = self.initial_penalty_coeff

        if not prob.find_closest_feasible_point():
            return False

        for i in range(self.max_merit_coeff_increases):
            success = self._min_merit_fn(
                prob, penalty_coeff, trust_region_size, verbose=verbose
            )
            # prob._update_vars()
            if verbose:
                print("\n")

            if prob.get_max_cnt_violation() > self.cnt_tolerance:
                penalty_coeff = penalty_coeff * self.merit_coeff_increase_ratio
                trust_region_size = self.initial_trust_region_size
            else:
                end = time.time()
                if verbose:
                    print("sqp time: ", end - start)
                return success
        end = time.time()
        if verbose:
            print("sqp time: ", end - start)
        return False

    # @profile
    def _min_merit_fn(self, prob, penalty_coeff, trust_region_size, verbose=False):
        """
        Returns true if the merit function is minimized successfully.
        Minimize merit function for penalty sqp
        """
        sqp_iter = 1

        while True:
            if verbose:
                print(("  sqp_iter: {0}".format(sqp_iter)))

            prob.convexify()
            prob.update_obj(penalty_coeff)
            merit = prob.get_value(penalty_coeff)
            merit_vec = prob.get_value(penalty_coeff, True)
            prob.save()

            while True:
                if verbose:
                    print(("    trust region size: {0}".format(trust_region_size)))

                prob.add_trust_region(trust_region_size)
                prob.optimize()

                model_merit = prob.get_approx_value(penalty_coeff)
                model_merit_vec = prob.get_approx_value(penalty_coeff, True)
                new_merit = prob.get_value(penalty_coeff)

                approx_merit_improve = merit - model_merit
                if not approx_merit_improve:
                   approx_merit_improve += 1e-10

                ## we converge if one of the violated constraint groups
                ## is below the minimum improvement
                approx_improve_vec = merit_vec - model_merit_vec
                violated = merit_vec > self.cnt_tolerance
                if approx_improve_vec.shape == (0,):
                    approx_improve_vec = np.array([approx_merit_improve])
                    violated = approx_improve_vec > -np.inf

                exact_merit_improve = merit - new_merit
                merit_improve_ratio = exact_merit_improve / approx_merit_improve

                if verbose:
                    print(
                        (
                            "      merit: {0}. model_merit: {1}. new_merit: {2}".format(
                                merit, model_merit, new_merit
                            )
                        )
                    )
                    print(
                        (
                            "      approx_merit_improve: {0}. exact_merit_improve: {1}. merit_improve_ratio: {2}".format(
                                approx_merit_improve,
                                exact_merit_improve,
                                merit_improve_ratio,
                            )
                        )
                    )

                if self._bad_model(approx_merit_improve):
                    if verbose:
                        print(
                            (
                                "Approximate merit function got worse ({0})".format(
                                    approx_merit_improve
                                )
                            )
                        )
                        print(
                            "Either convexification is wrong to zeroth order, or you're in numerical trouble."
                        )
                    prob.restore()
                    return False

                if self._y_converged(approx_merit_improve):
                    if verbose:
                        print("Converged: y tolerance")
                    prob.restore()
                    return True

                ## we converge if one of the violated constraint groups
                ## is below the minimum improvement and none of its overlapping
                ## groups are making progress
                prob.nonconverged_groups = []
                for gid, idx in prob.gid2ind.items():
                    if (
                        violated[idx]
                        and approx_improve_vec[idx] < self.min_approx_improve
                    ):
                        overlap_improve = False
                        for gid2 in prob._cnt_groups_overlap[gid]:
                            if (
                                approx_improve_vec[prob.gid2ind[gid2]]
                                > self.min_approx_improve
                            ):
                                overlap_improve = True
                                break
                        if overlap_improve:
                            continue
                        prob.nonconverged_groups.append(gid)
                if len(prob.nonconverged_groups) > 0:
                    if verbose:
                        print("Converged: y tolerance")
                    prob.restore()
                    ## store the failed groups into the prob

                    for i, g in enumerate(sorted(prob._cnt_groups.keys())):
                        if violated[i] and self._y_converged(approx_improve_vec[i]):
                            prob.nonconverged_groups.append(g)
                    return True

                if self._shrink_trust_region(exact_merit_improve, merit_improve_ratio):
                    prob.restore()
                    if verbose:
                        print("Shrinking trust region")
                    trust_region_size = trust_region_size * self.trust_shrink_ratio
                else:
                    if verbose:
                        print("Growing trust region")
                    trust_region_size = trust_region_size * self.trust_expand_ratio
                    break  # from trust region loop

                if self._x_converged(trust_region_size):
                    if verbose:
                        print("Converged: x tolerance")
                    return True

            sqp_iter = sqp_iter + 1


    def _bad_model(self, approx_merit_improve):
        """
        Returns true if the approx_merit_improve is too low which suggests that
        either the convexification is wrong to the zeroth order or there are
        numerical problems.
        """
        return approx_merit_improve < -1e-5

    def _shrink_trust_region(self, exact_merit_improve, merit_improve_ratio):
        """
        Returns true if the trust region should shrink (exact merit improve is negative or the merit improve ratio is too low)
        """
        return (exact_merit_improve < 0) or (
            merit_improve_ratio < self.improve_ratio_threshold
        )

    def _x_converged(self, trust_region_size):
        """
        Returns true if the variable values has converged (trust_region size is
        smaller than the minimum trust region size)
        """
        return trust_region_size < self.min_trust_region_size

    def _y_converged(self, approx_merit_improve):
        """
        Returns true if the approx_merit has converged (approx_merit_improve <
        min_approx_merit_improve)
        """
        return approx_merit_improve < self.min_approx_improve
