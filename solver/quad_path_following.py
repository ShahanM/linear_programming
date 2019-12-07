import numpy as np
from solver.abs_solver import AbstractSolver

# FIXME The loop break condition is broken, check math


class QuadraticPathFollowing(AbstractSolver):
    def __init__(self, hessian_mat, coefficient_mat, constraint_vec, kkt_coefficient, init_val=0.9, iter_data=True):
        super().__init__(coefficient_mat, constraint_vec, kkt_coefficient, init_val, iter_data)
        self.Q = hessian_mat
        self.A = coefficient_mat
        self.b = constraint_vec
        self.c = kkt_coefficient
        self.init_val = init_val

        # TODO add the iteration data

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

        # FIXME this condition does not work
        while np.dot(x, s) > epsilon:
            k += 1

            mu_k = (np.dot(x, s) + np.dot(u, v)) / (m + n)
            # row 2
            jacobian[n+m:n+m+n, 0:n] = np.diag(s)
            jacobian[n+m:n+m+n, n:n+n] = np.diag(x)

            # row 3
            jacobian[n+m+n:n+m+n+m, n+n:n+n+m] = np.diag(v)
            jacobian[n+m+n:n+m+n+m, n+n+m:n+n+m+m] = np.diag(u)

            newton[0:n] = np.dot(self.A.T, u) + s - self.c + np.dot(self.Q, x)
            newton[n:n+m] = v + self.b - np.dot(self.A, x)

            e_n = np.ones(shape=(n, ), dtype=np.float64)
            newton[n+m:n+m+n] = sigma*mu_k*e_n - np.dot(np.diag(x), np.dot(np.diag(s), e_n))

            e_m = np.ones(shape=(m, ), dtype=np.float64)
            newton[n+m+n:n+m+n+m] = sigma*mu_k*e_m - np.dot(np.diag(u), np.dot(np.diag(v), e_m))

            delta = np.linalg.solve(jacobian, newton)
            delta_x = delta[0:n]
            delta_u = delta[n:n + m]
            delta_s = delta[n+m:n+m+n]
            delta_v = delta[n+m+n:n+m+n+m]

            alpha_x, alpha_s, alpha_k_xs = self._get_steplength(x, s, delta_x, delta_s, eta)
            alpha_u, alpha_v, alpha_k_uv = self._get_steplength(u, v, delta_u, delta_v, eta)

            alpha_k = min(1.0, eta * min(alpha_k_xs, alpha_k_uv))

            # create new iterate
            x = x + alpha_k * delta_x
            s = s + alpha_k * delta_s
            u = u + alpha_k * delta_u
            v = v + alpha_k * delta_v

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
    x = qp_solver.solve()

    print(x)
