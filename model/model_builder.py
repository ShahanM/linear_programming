import numpy as np


class LinearProgrammingModel:
    def __init__(self, num_vars, eq_constraints=None, gt_constraints=None, lt_constraints=None,
                 lt_constraints_val=None, gt_constraints_val=None, eq_constraints_val=None):
        if eq_constraints is None:
            eq_constraints = []
        if gt_constraints is None:
            gt_constraints = []
        if lt_constraints is None:
            lt_constraints = []
        if lt_constraints_val is None:
            self.lt_constraints_val = [0] * len(lt_constraints)
        else:
            self.lt_constraints_val = lt_constraints_val
        if gt_constraints_val is None:
            self.gt_constraints_val = [0] * len(gt_constraints)
        else:
            self.gt_constraints_val = gt_constraints_val
        if eq_constraints_val is None:
            self.eq_constraints_val = [0] * len(eq_constraints)
        else:
            self.eq_constraints_val = eq_constraints_val
        self.num_vars = num_vars
        self.eq_constraints = eq_constraints
        self.gt_constraints = gt_constraints
        self.lt_constraints = lt_constraints
        self.A = np.zeros(shape=(len(gt_constraints) + len(eq_constraints) + len(lt_constraints),
                                 num_vars + len(gt_constraints) + len(lt_constraints)),
                          dtype=np.float64)

    def get_constraint_matrix(self):
        self._fill_constraint_matrix(0, self.eq_constraints, 'eq')
        self._fill_constraint_matrix(len(self.eq_constraints), self.lt_constraints, 'lt')
        self._fill_constraint_matrix(len(self.eq_constraints) + len(self.lt_constraints),
                                     self.gt_constraints, 'gt')

        return self.A

    def get_constraint_vector(self):
        b = []
        b.extend(self.eq_constraints_val)
        b.extend(self.lt_constraints_val)
        b.extend(self.gt_constraints_val)

        return np.asarray(b)

    def _fill_constraint_matrix(self, start_row, constraints_tuple, constraint_type):
        for row in range(len(constraints_tuple)):
            effective_row = row + start_row
            for (idx, coeff) in constraints_tuple[row]:
                self.A[effective_row, idx] = coeff
            if constraint_type == 'lt':
                self.A[effective_row, self.num_vars] = 1
                self.num_vars += 1
            if constraint_type == 'gt':
                self.A[effective_row, self.num_vars] = -1
                self.num_vars += 1
