import numpy as np
from solver.quadratic_solver import QPSolver


class QPPathFollowing(QPSolver):
    def __init__(self, hessian_mat, coefficient_mat, constraint_vec, kkt_coefficient, init_val=0.9, iter_data=True):
        super().__init__(hessian_mat, coefficient_mat, constraint_vec, kkt_coefficient, init_val, iter_data)

    def solve(self, sigma=0.95, eta=0.95, epsilon=0.0001):
        m = self.A.shape[0]
        n = self.A.shape[1]
        x, s = self._init_vec(n, 2)
        u, v = self._init_vec(m, 2)

        red_jac = np.zeros(shape=(n + m, n + m))
        red_new = np.zeros(shape=(n + m,))

        k = 0

        while np.dot(x, s) > epsilon and np.dot(u, v) > epsilon:

            k += 1

            mu_k = (np.dot(x, s) + np.dot(u, v)) / (m + n)

            x_reciprocal = np.diag(np.reciprocal(x))
            u_reciprocal = np.diag(np.reciprocal(u))

            red_jac[0:n, 0:n] = np.copy(self.Q + np.dot(x_reciprocal, np.diag(s)))
            red_jac[0:n, n:n + m] = np.copy(-self.A.T)
            red_jac[n:n + m, 0:n] = np.copy(-self.A)
            red_jac[n:n + m, n:n + m] = np.copy(-np.dot(u_reciprocal, np.diag(v)))

            rp = np.dot(self.A.T, u) + s - self.c - np.dot(self.Q, x)
            rb = v + self.b - np.dot(self.A, x)

            e_n = np.ones(shape=(n, ), dtype=np.float64)
            rxs = sigma*mu_k*e_n - np.dot(np.diag(x), np.dot(np.diag(s), e_n))

            e_m = np.ones(shape=(m, ), dtype=np.float64)
            ruv = sigma*mu_k*e_m - np.dot(np.diag(u), np.dot(np.diag(v), e_m))

            red_new[0:n] = rp + np.dot(x_reciprocal, rxs)
            red_new[n:n + m] = -rb - np.dot(u_reciprocal, ruv)

            delta = np.linalg.solve(red_jac, red_new)
            delta_x = delta[0:n]
            delta_u = delta[n:n + m]

            delta_s = np.dot(x_reciprocal, rxs - np.dot(np.diag(s), delta_x))
            delta_v = np.dot(u_reciprocal, ruv - np.dot(np.diag(v), delta_u))

            x, s, u, v = self._update_data(x, s, u, v, delta_x, delta_s, delta_u, delta_v, eta, k)
        return x


if __name__ == "__main__":
    A = np.asarray(np.asmatrix('1 -2; \
                         -1 -2; \
                         -1 2'), dtype=np.float64)
    b = np.asarray([-2, -6, -2], dtype=np.float64)
    c = np.asarray([-2, -5], dtype=np.float64)

    Q = np.asarray(np.asmatrix('4 0; \
                         0 4'), dtype=np.float64)

    qp_solver = QPPathFollowing(Q, A, b, c, 10e-3)
    X = qp_solver.solve(epsilon=1e-6)

    print('Main Result {}'.format(X))
