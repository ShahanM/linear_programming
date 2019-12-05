import numpy as np
from solver.interior_point import InteriorPointMethod
from solver.path_following import PathFollowing

if __name__ == "__main__":
    A = np.asarray(np.asmatrix('1, 0, 1, 0, 0, 0, 0; \
                         2, 2, 0, 1, 0, 0, 0; \
                         4, 1, 0, 0, 1, 0, 0; \
                         4, 2, 0, 0, 0, 1, 0; \
                         1, 2.2, 0, 0, 0, 0, 1'), dtype=np.float64)
    b = np.asarray([2.3, 10, 10, 12, 10], dtype=np.float64)
    c = np.asarray([-1, -2, 0, 0, 0, 0, 0], dtype=np.float64)

    epsilon = 1e-3

    ipm = InteriorPointMethod(A, b, c)
    x_ipm, x_imp_iter = ipm.solve(theta=0.95, gamma=0.01, epsilon=epsilon)
    print(ipm.print_metric())

    cpm = PathFollowing(A, b, c)
    x_cpm, x_cpm_iter = cpm.solve(eta=0.95, sigma=0.1, epsilon=epsilon)
    print(cpm.print_metric())
