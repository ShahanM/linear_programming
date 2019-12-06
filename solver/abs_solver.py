import numpy as np
from abc import abstractmethod
from collections import OrderedDict


class AbstractSolver:
    def __init__(self, coefficient_mat, constraint_vec, minimization_coefficient, init_val, iter_data):
        assert coefficient_mat.shape[0] == constraint_vec.shape[0], \
            'dimensions of A and b must be compatible!'
        assert coefficient_mat.shape[1] == minimization_coefficient.shape[0], \
            'dimensions of A and c must be compatible!'
        self.mat_a = coefficient_mat
        self.vec_b = constraint_vec
        self.vec_c = minimization_coefficient

        self.init_val = init_val
        if iter_data:
            self.iter_metric = OrderedDict()

    @abstractmethod
    def solve(self):
        pass

    def _init_x_s(self):
        n = self.mat_a.shape[1]

        x = np.empty(shape=(n, ), dtype=np.float64)
        s = np.empty(shape=(n, ), dtype=np.float64)

        x.fill(self.init_val)
        s.fill(self.init_val)

        return x, s

    def _get_alpha(self, x, s, delta_x, delta_s, theta):
        alpha_x = 1.0
        alpha_s = 1.0
        for i in range(self.mat_a.shape[1]):
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

        return alpha_k, alpha_x, alpha_s

    def _gen_data(self, k, x, y, s, delta_x, delta_y, delta_s, alpha_k):
        iter_data = {}
        diff = np.dot(self.mat_a, x) - self.vec_b
        iter_data['iter_num'] = k
        iter_data['diff'] = diff
        iter_data['diff_norm'] = np.linalg.norm(diff)
        iter_data['delta_x'] = delta_x
        iter_data['delta_y'] = delta_y
        iter_data['delta_s'] = delta_s
        iter_data['alpha_k'] = alpha_k
        iter_data['x_val'] = x
        iter_data['y_val'] = y
        iter_data['s_val'] = s
        iter_data['obj_val'] = np.dot(self.vec_c, x)

        self.iter_metric[k] = iter_data

    def get_metric(self):
        if self.iter_metric is None:
            return {}
        else:
            return self.iter_metric

    def print_metric(self):
        if self.iter_metric is None:
            print('Iteration data was not recorded.')
        else:
            for k, v in self.iter_metric.items():
                print('=====================================================')
                print('Iteration -> {}'.format(k))
                print('-----------------------------------------------------')
                print('Delta Y -> {}'.format(v['delta_y']))
                print('-----------------------------------------------------')
                print('Delta S -> {}'.format(v['delta_s']))
                print('-----------------------------------------------------')
                print('Delta X -> {}'.format(v['delta_x']))
                print('-----------------------------------------------------')
                print('Ax - b = {} '.format(v['diff']))
                print('-----------------------------------------------------')
                print('Norm of Ax - b -> {} '.format(v['diff_norm']))
                print('-----------------------------------------------------')
                print('Coordinates -> {}'.format(v['x_val']))
                print('-----------------------------------------------------')
                print('Functional value -> {}'.format(v['obj_val']))
                print('=====================================================')
