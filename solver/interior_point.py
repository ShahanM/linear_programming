import numpy as np

from solver.abs_solver import AbstractSolver


class InteriorPointMethod(AbstractSolver):
    def __init__(self, coefficient_mat, constraint_vec, minimization_coefficient, init_val=0.9, iter_data=True):
        super().__init__(coefficient_mat, constraint_vec, minimization_coefficient, init_val, iter_data)

    def solve(self, theta=0.95, gamma=0.1, epsilon=0.0001):
        n = self.mat_a.shape[1]

        # Set initial points of (x, y, s), where x > 0 and s > 0
        x, s, e = self._init_vec(n, 3)

        y = np.zeros(shape=(self.mat_a.shape[0],))

        k = 0

        if self.iter_metric is not None:
            self.__gen_data(k, x, y, s, 0, 0, 0, 1.0, 1.0, 1.0, None, None)

        x_iterations = []

        while np.dot(x, s) > epsilon:

            x_iterations.append(x)

            r_primal = self.vec_b - np.dot(self.mat_a, x)
            r_dual = self.vec_c - np.dot(self.mat_a.T, y) - s

            mu_k = np.dot(x, s) / n

            s_reciprocal = np.diag(np.reciprocal(s))
            x_ = np.diag(x)

            m_ = self.mat_a.dot(s_reciprocal).dot(x_).dot(self.mat_a.T)
            r = self.vec_b + self.mat_a.dot(s_reciprocal).dot(x_.dot(r_dual) - (gamma * mu_k * e))

            delta_y = np.linalg.solve(m_, r)
            delta_s = r_dual - np.dot(self.mat_a.T, delta_y)
            delta_x = np.dot(s_reciprocal, (gamma * mu_k * e) - np.dot(x_, delta_s)) - x

            alpha_k, alpha_x, alpha_s = self._get_steplength(x, s, delta_x, delta_s, theta)

            # create new iterate
            x = x + alpha_x * delta_x
            y = y + alpha_k * delta_y
            s = s + alpha_s * delta_s

            k += 1

            if self.iter_metric is not None:
                self.__gen_data(k, x, y, s, delta_x, delta_y, delta_s, alpha_k, alpha_x, alpha_s, m_, r)

        return x, x_iterations

    def __gen_data(self, k, x, y, s, delta_x, delta_y, delta_s, alpha_k, alpha_x, alpha_s, m, r):
        super()._gen_data(k, x, y, s, delta_x, delta_y, delta_s, alpha_k)
        self.iter_metric[k]['alpha_x'] = alpha_x
        self.iter_metric[k]['alpha_s'] = alpha_s
        self.iter_metric[k]['m'] = m
        self.iter_metric[k]['r'] = r
