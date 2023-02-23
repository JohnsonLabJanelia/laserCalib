import numpy as np

def get_inverse_transformation(rotation_matrix, translation):
    r_inv = rotation_matrix.T    
    t_inv = -np.matmul(rotation_matrix.T, translation)
    return r_inv, t_inv