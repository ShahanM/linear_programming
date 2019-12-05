import numpy as np

from solver.abs_solver import AbstractSolver


class InteriorPointMethod(AbstractSolver):
    def __init__(self, coefficient_mat, constraint_vec, minimization_coefficient, init_val=0.9, iter_data=True):
        super().__init__(coefficient_mat, constraint_vec, minimization_coefficient, init_val, iter_data)

    def solve(self, theta=0.95, gamma=0.1, epsilon=0.0001):
        n = self.mat_a.shape[1]
        # Set initial points of (x, y, s), where x > 0 and s > 0
        x = np.empty(shape=(n, ), dtype=np.float64)
        s = np.empty(shape=(n, ), dtype=np.float64)
        e = np.empty(shape=(n, ), dtype=np.float64)
        x.fill(self.init_val)
        s.fill(self.init_val)
        e.fill(self.init_val)

        y = np.zeros(shape=(self.mat_a.shape[0], ))

        k = 0

        if self.iter_metric is not None:
            self.__gen_data(k, x, y, s, 0, 0, 0, 1.0, None, None)

        x_iterations = []

        while np.dot(x.T, s) > epsilon:

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

            alpha_x = 1.0
            alpha_s = 1.0
            for i in range(n):
                if delta_x[i] < 0:
                    # get alpha primal
                    alpha_x = min(alpha_x, -x[i]/delta_x[i])
                if delta_s[i] < 0:
                    # get alpha dual
                    alpha_s = min(alpha_s, -s[i]/delta_s[i])
            # choose smallest alpha
            alpha_x = min(1.0, theta * alpha_x)
            alpha_s = min(1.0, theta * alpha_s)
            alpha_k = min(1.0, theta * min(alpha_s, alpha_x))

            # create new iterate
            x = x + alpha_x * delta_x
            y = y + alpha_k * delta_y
            s = s + alpha_s * delta_s

            if self.iter_metric is not None:
                self.__gen_data(k, x, y, s, delta_x, delta_y, delta_s, alpha_k, m_, r)

            k += 1

        return x, x_iterations

    def __gen_data(self, k, x, y, s, delta_x, delta_y, delta_s, alpha_k, m, r):
        super()._gen_data(k, x, y, s, delta_x, delta_y, delta_s, alpha_k)
        self.iter_metric[k]['m'] = m
        self.iter_metric[k]['r'] = r
