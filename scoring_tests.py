import numpy

import peas
d = numpy.arange(25).reshape((5,5))

print('sum')
print(peas.scoring_funcs_cython.compute_sum_table_2d(d.astype(float), 0, 4))

print('shuffled sum')
print(peas.scoring_funcs_cython.compute_sum_table_2d_shuffled(d.astype(float), 0, 4, 1))

# print('mean')
# print(peas.scoring_funcs_cython.compute_mean_table_2d(d.astype(float), 0, 4))


print('shuffled_mean')
print(peas.scoring_funcs_cython.compute_mean_table_2d_shuffled(d.astype(float), 0, 4, 1))
