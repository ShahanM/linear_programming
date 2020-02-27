import numpy as np
from solver.quadratic_solver import QPSolver


class QuadraticPathFollowing(QPSolver):
    def __init__(self, hessian_mat, coefficient_mat, constraint_vec, kkt_coefficient, init_val=0.9, iter_data=True):
        super().__init__(hessian_mat, coefficient_mat, constraint_vec, kkt_coefficient, init_val, iter_data)

    def solve(self, sigma=0.95, eta=0.95, epsilon=0.0001):
        m = self.A.shape[0]
        n = self.A.shape[1]
        x, s = self._init_vec(n, 2)
        u, v = self._init_vec(m, 2)

        jacobian = np.zeros(shape=(n + m + n + m, n + n + m + m))
        # row 0
        jacobian[0:n, 0:n] = np.copy(self.Q)
        jacobian[0:n, n:n+n] = np.eye(n)
        jacobian[0:n, n+n:n+n+m] = np.copy(-self.A.T)
        # row 1
        jacobian[n:n+m, 0:n] = np.copy(self.A)
        jacobian[n:n+m, n+n+m:n+n+m+m] = -np.eye(m)

        newton = np.zeros(shape=(n + m + n + m,))

        k = 0

        while np.dot(x, s) > epsilon and np.dot(u, v) > epsilon:
            k += 1

            mu_k = (np.dot(x, s) + np.dot(u, v)) / (m + n)

            # row 2
            jacobian[n+m:n+m+n, 0:n] = np.diag(s)
            jacobian[n+m:n+m+n, n:n+n] = np.diag(x)

            # row 3
            jacobian[n+m+n:n+m+n+m, n+n:n+n+m] = np.diag(v)
            jacobian[n+m+n:n+m+n+m, n+n+m:n+n+m+m] = np.diag(u)

            newton[0:n] = np.dot(self.A.T, u) + s - self.c - np.dot(self.Q, x)
            newton[n:n+m] = v + self.b - np.dot(self.A, x)

            e_n = np.ones(shape=(n, ), dtype=np.float64)
            newton[n+m:n+m+n] = sigma*mu_k*e_n - np.dot(np.diag(x), np.dot(np.diag(s), e_n))

            e_m = np.ones(shape=(m, ), dtype=np.float64)
            newton[n+m+n:n+m+n+m] = sigma*mu_k*e_m - np.dot(np.diag(u), np.dot(np.diag(v), e_m))

            delta = np.linalg.solve(jacobian, newton)

            delta_x = delta[:n]
            delta_s = delta[n:n + n]
            delta_u = delta[n + n: n + n + m]
            delta_v = delta[n + n + m:]

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

    qp_solver = QuadraticPathFollowing(Q, A, b, c)
    X = qp_solver.solve(epsilon=1e-6)

    print('Main Result {}'.format(X))
