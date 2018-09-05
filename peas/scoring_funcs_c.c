#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdlib.h>
#include <stdio.h>
#include "my_array_funcs.h"

// #define printf PySys_WriteStdout


void c_compute_sum_table_2d(double* data, size_t matrix_size, size_t start_diagonal, size_t end_diagonal, double* sum_table){
    /*
    Returns an upper-triangular matrix where each cell contains the sum of a square
    subset of :param:`data`centered on the diagonal with a corner in that cell, excluding
    the diagonal itself.

    Uses implicit recursion to do this efficiently.
    */
    size_t col_idx;
	double *sum_table;
	double **sum_rows;
	double **data_rows;

	sum_table = (double*) calloc(matrix_size * matrix_size, sizeof(double));
	sum_rows = get_row_ptrs(sum_table, matrix_size, matrix_size);
	data_rows = get_row_ptrs(data, matrix_size, matrix_size);

    for (k = start_diagonal; k <= end_diagonal; k++){
        for (row_idx = 0; row_idx <= end_diagonal - k; row_idx++){
            sum_rows[row_idx][col_idx] = 0;
            col_idx = row_idx + k;
            // current cell
            sum_rows[row_idx][col_idx] += data_rows[row_idx][col_idx];

            if k - start_diagonal >= 1:
                // left cell
                sum_rows[row_idx][col_idx] += sum_rows[row_idx][col_idx - 1];
                // beneath cell
                sum_rows[row_idx][col_idx] += sum_rows[row_idx + 1][col_idx];

                if k - start_diagonal >= 2:
                    sum_rows[row_idx][col_idx] -= sum_rows[row_idx + 1][col_idx - 1];

        }
    }
    free(col_idx);
    free(sum_table);
    free(data_rows);
    return;
}
