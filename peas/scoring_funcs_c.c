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

    Uses implicit recursion to do this efficiently..

    Note that :param end_diagonal: follows Python convention and will _not_ be included in the
    result.
    */
    size_t row_idx, col_idx, k;
	double **sum_rows; // pointer arrays for convenient indexing
	double **data_rows;

//	sum_table = (double*) calloc(matrix_size * matrix_size, sizeof(double));
	sum_rows = get_row_ptrs(sum_table, matrix_size, matrix_size);
	data_rows = get_row_ptrs(data, matrix_size, matrix_size);

    for (k = start_diagonal; k < end_diagonal; k++){
        for (row_idx = 0; row_idx < matrix_size - k; row_idx++){
//            printf("%zu", row_idx);
            col_idx = row_idx + k;
            sum_rows[row_idx][col_idx] = 0;
            // current cell
            sum_rows[row_idx][col_idx] += data_rows[row_idx][col_idx];

            if (k - start_diagonal >= 1){
                // left cell
                sum_rows[row_idx][col_idx] += sum_rows[row_idx][col_idx - 1];
                // beneath cell
                sum_rows[row_idx][col_idx] += sum_rows[row_idx + 1][col_idx];

                if (k - start_diagonal >= 2){
                    sum_rows[row_idx][col_idx] -= sum_rows[row_idx + 1][col_idx - 1];
                }
            }
        }
    }
//    print_matrix(sum_rows, matrix_size, matrix_size);
    free(data_rows);
    free(sum_rows);
    return;
}


void c_compute_sum_table_2d_shuffled(double* data, size_t matrix_size, size_t start_diagonal, size_t end_diagonal, double* sum_table){
    /*
    Returns an upper-triangular matrix where each cell contains the sum of a square
    subset of :param:`data`centered on the diagonal with a corner in that cell, excluding
    the diagonal itself.

    Uses implicit recursion to do this efficiently..

    Note that :param end_diagonal: follows Python convention and will _not_ be included in the
    result.
    */
    size_t row_idx, col_idx, k;
	double **sum_rows; // pointer arrays for convenient indexing
	double **data_rows;
	double this_cell;

    size_t shuffled_index[matrix_size];
    for (size_t i = 0; i < matrix_size; i++){
        shuffled_index[i] = i;
    }
    shuffle_array_st(shuffled_index, matrix_size);
//    print_vec_l(shuffled_index, matrix_size);

	sum_rows = get_row_ptrs(sum_table, matrix_size, matrix_size);
	data_rows = get_row_ptrs(data, matrix_size, matrix_size);

    for (k = start_diagonal; k < end_diagonal; k++){
        for (row_idx = 0; row_idx < matrix_size - k; row_idx++){
//            this_cell = 0;
//            printf("%zu", row_idx);
            col_idx = row_idx + k;
            // current cell
//            printf("row %zu col %zu\n", row_idx, col_idx);
//            printf("row_idx %zu col_idx %zu\n", shuffled_index[row_idx], shuffled_index[col_idx]);
            this_cell = data_rows[shuffled_index[row_idx]][shuffled_index[col_idx]];

            if (k - start_diagonal >= 1){
                // left cell
                this_cell += sum_rows[row_idx][col_idx - 1];
                // beneath cell
                this_cell += sum_rows[row_idx + 1][col_idx];

                if (k - start_diagonal >= 2){
                    this_cell -= sum_rows[row_idx + 1][col_idx - 1];
                }
            }
            sum_rows[row_idx][col_idx] = this_cell;
        }
    }
//    print_matrix(sum_rows, matrix_size, matrix_size);
    free(data_rows);
    free(sum_rows);
//    free(shuffled_index);
    return;
}


void generate_2d_denominator_table(size_t n, size_t start_diagonal, size_t end_diagonal, double* denominator)
{
    double** denom_rows = get_row_ptrs(denominator, n, n);

    double cumulant = 0;

    for (size_t diag_idx = start_diagonal; diag_idx < n; diag_idx++)
    {
        cumulant += diag_idx - start_diagonal + 1;
        for (size_t row_idx = 0; row_idx < n - diag_idx; row_idx++)
        {
            size_t col_idx = row_idx + diag_idx;
            denom_rows[row_idx][col_idx] = cumulant;
        }
    }
    return;
}


void c_compute_mean_table_2d_shuffled(double* data, size_t matrix_size, size_t start_diagonal, size_t end_diagonal, double* mean_table)
{
    /*
    Returns an upper-triangular matrix where each cell contains the sum of a square
    subset of :param:`data`centered on the diagonal with a corner in that cell, excluding
    the diagonal itself.

    Uses implicit recursion to do this efficiently..

    Note that :param end_diagonal: follows Python convention and will _not_ be included in the
    result.
    */
    double** mean_rows = get_row_ptrs(mean_table, matrix_size, matrix_size);
    c_compute_sum_table_2d_shuffled(data, matrix_size, start_diagonal, end_diagonal, mean_table);
    double cumulant = 0;

    for (size_t diag_idx = start_diagonal; diag_idx < end_diagonal; diag_idx++)
    {
        cumulant += diag_idx - start_diagonal + 1;
        for (size_t row_idx = 0; row_idx < matrix_size - diag_idx; row_idx++)
        {
            size_t col_idx = row_idx + diag_idx;
            mean_rows[row_idx][col_idx] /= cumulant;
        }
    }
    free(mean_rows);
    return;
}


void c_compute_sum_table_1d_shuffled(double* data, size_t vector_length, size_t end_diagonal, double* sum_table){
    /*
    Returns an upper-triangular matrix where each cell contains the sum of a
    subset of :param:`data`from :param row: to :param col:.

    Uses implicit recursion to do this efficiently..

    Note that :param end_diagonal: follows Python convention and will _not_ be included in the
    result.
    */
    size_t row_idx, col_idx, k;
	double **sum_rows; // pointer arrays for convenient indexing
	double this_cell;

    size_t shuffled_index[vector_length];
    for (size_t i = 0; i < vector_length; i++){
        shuffled_index[i] = i;
    }
    shuffle_array_st(shuffled_index, vector_length);

	sum_rows = get_row_ptrs(sum_table, vector_length, vector_length);

    for (k = 0; k < end_diagonal; k++){
        for (row_idx = 0; row_idx < vector_length - k; row_idx++){
           this_cell = 0;
            col_idx = row_idx + k;

            if (k == 0){
				this_cell = data[shuffled_index[row_idx]];
			}
			else {
				this_cell += sum_rows[row_idx][col_idx - 1];
				this_cell += data[shuffled_index[col_idx]];
			
			}
            sum_rows[row_idx][col_idx] = this_cell;
        }
    }
    free(sum_rows);
    return;
}	


void c_compute_mean_table_1d_shuffled(double* data, size_t vector_length, size_t end_diagonal, double* mean_table)
{
    /*
    Returns an upper-triangular matrix where each cell contains the sum of a square
    subset of :param:`data`centered on the diagonal with a corner in that cell, excluding
    the diagonal itself.

    Uses implicit recursion to do this efficiently..

    Note that :param end_diagonal: follows Python convention and will _not_ be included in the
    result.
    */
    double** mean_rows = get_row_ptrs(mean_table, vector_length, vector_length);
    c_compute_sum_table_1d_shuffled(data, vector_length, end_diagonal, mean_table);
    // double cumulant = 0;

    for (size_t diag_idx = 0; diag_idx < end_diagonal; diag_idx++)
    {
        // cumulant += diag_idx - 0 + 1;
        for (size_t row_idx = 0; row_idx < vector_length - diag_idx; row_idx++)
        {
            size_t col_idx = row_idx + diag_idx;
            mean_rows[row_idx][col_idx] /= (diag_idx + 1);
        }
    }
    free(mean_rows);
    return;
}
