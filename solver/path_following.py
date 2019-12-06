import numpy as np
from solver.abs_solver import AbstractSolver


class PathFollowing(AbstractSolver):
    def __init__(self, coefficient_mat, constraint_vec, minimization_coefficient, init_val=0.9, iter_data=True):
        super().__init__(coefficient_mat, constraint_vec, minimization_coefficient, init_val, iter_data)

    def solve(self, eta=0.95, sigma=0.1, epsilon=0.0001):
        m = self.mat_a.shape[0]
        n = self.mat_a.shape[1]

        x, s = self._init_x_s()

        y = np.empty(shape=(m,), dtype=np.float64)
        y.fill(self.init_val)

        jacobian = np.zeros(shape=(m + n + n, n + m + n))
        jacobian[0:m, 0:n] = np.copy(self.mat_a)
        jacobian[m:m + n, n:n + m] = np.copy(self.mat_a.T)
        jacobian[m:m + n, n + m:n + m + n] = np.eye(n)

        newton = np.zeros(shape=(n + m + n,))

        k = 0

        if self.iter_metric is not None:
            self._gen_data(k, x, y, s, 0, 0, 0, 1.0)

        x_iterations = []
        sigma_k = sigma
        while abs(np.dot(x, s)) > epsilon:
            k += 1

            mu_k = np.dot(x, s) / n

            jacobian[m + n:m + n + n, 0:n] = np.diag(s)
            jacobian[m + n:m + n + n, n + m:n + m + n] = np.diag(x)

            newton[0:m] = self.vec_b - np.dot(self.mat_a, x)
            newton[m:m + n] = self.vec_c - np.dot(self.mat_a.T, y) - s
            newton[m + n:m + n + n] = np.copy(sigma_k * mu_k * np.ones(shape=(n,))
                                              - np.dot(np.dot(np.diag(x), np.diag(s)), np.ones(shape=(n,))))

            # solve for delta
            delta = np.linalg.solve(jacobian, newton)
            delta_x = delta[0:n]
            delta_y = delta[n:n + m]
            delta_s = delta[n + m:n + m + n]

            alpha_k, alpha_x, alpha_s = self._get_alpha(x, s, delta_x, delta_s, eta)

            x = x + alpha_k * delta_x
            y = y + alpha_k * delta_y
            s = s + alpha_k * delta_s

            if self.iter_metric is not None:
                self._gen_data(k, x, y, s, delta_x, delta_y, delta_s, alpha_k)

            x_iterations.append(x)

        return x, x_iterations
