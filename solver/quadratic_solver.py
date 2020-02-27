from abc import abstractmethod

import numpy as np
from solver.abs_solver import AbstractSolver


class QPSolver(AbstractSolver):
    def __init__(self, hessian_mat, coefficient_mat, constraint_vec, kkt_coefficient, init_val=0.9, iter_data=True):
        super().__init__(coefficient_mat, constraint_vec, kkt_coefficient, init_val, iter_data)
        self.Q = hessian_mat
        self.A = coefficient_mat
        self.b = constraint_vec
        self.c = kkt_coefficient

    def _update_data(self, x, s, u, v, delta_x, delta_s, delta_u, delta_v, eta, k):
        alpha_x, alpha_s, alpha_k_xs = self._get_steplength(x, s, delta_x, delta_s, eta)
        alpha_u, alpha_v, alpha_k_uv = self._get_steplength(u, v, delta_u, delta_v, eta)

        print('Iteration -> {}'.format(k))

        print('delta x -> {}'.format(delta_x))
        print('delta s -> {}'.format(delta_s))
        print('delta u -> {}'.format(delta_u))
        print('delta v -> {}'.format(delta_v))

        print('alpha x -> {}'.format(alpha_x))
        print('alpha s -> {}'.format(alpha_s))
        print('alpha u -> {}'.format(alpha_u))
        print('alpha v -> {}'.format(alpha_v))

        # create new iterate
        x = x + alpha_x * delta_x
        s = s + alpha_s * delta_s
        u = u + alpha_u * delta_u
        v = v + alpha_v * delta_v

        print(x)

        print('XS -> {}'.format(np.dot(x, s)))
        print('UV -> {}'.format(np.dot(u, v)))

        return x, s, u, v
