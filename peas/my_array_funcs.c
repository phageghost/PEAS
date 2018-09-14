#include <stdlib.h>	
#include <stdio.h>
//#include <cblas.h>
// #include <time.h>
#include "my_array_funcs.h"

// #define printf PySys_WriteStdout


void arange(size_t start, size_t end, size_t* dest_vec){
    /*
    Analogous to numpy.arange, populates dest_vec with the integers
    in the half-open interval [start,end)
    */
    size_t sequence_length = end - start;
    size_t i;
    for (i = 0; i < sequence_length; i++){
        dest_vec[i] = start + i;
    }
    return;
}


size_t my_diag_indices(size_t n, size_t k, size_t* x_coords, size_t* y_coords){
    /*
    Return the indices corresponding to the kth diagonal of an n X n array
    in the form of a tuple of (x coords, y coords).

    Created since numpy does not provide this function.
    */
    size_t diag_size = n - abs(k);

    x_coords = calloc(diag_size, sizeof(size_t));
    y_coords = calloc(diag_size, sizeof(size_t));

    if (k <= 0){
        arange(-k, n, x_coords);
        arange(0, n + k, y_coords);
    }
    else{
        arange(0, n - k, x_coords);
        arange(k, n, y_coords);
    }
    return diag_size;
}


size_t unravel_index(size_t row_idx, size_t col_idx, size_t num_cols){
	return row_idx * num_cols + col_idx;
}


void* get_row_ptrs(double* arr, size_t num_rows, size_t num_cols){
    size_t row_idx;
    double** row_arr;
    
    row_arr = calloc(num_rows, sizeof(double*));
    
    for (row_idx = 0; row_idx < num_rows; row_idx++){
        row_arr[row_idx] = &arr[row_idx * num_cols];
    }
    return row_arr;
}


void copy_vec(double* source_vec, double* dest_vec, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		dest_vec[element_idx] = source_vec[element_idx];
	}
}


double dot_product_vec_vec(double* vec_a, double* vec_b, size_t vec_length){
	double accumulator = 0.0;
	size_t element_idx;
	
	for (element_idx = 0; element_idx < vec_length; element_idx++){
		accumulator += vec_a[element_idx] * vec_b[element_idx];
	}
	
	return accumulator;
}

//void openblas_dot_matrix_vec(double* arr, double* vec, double* result, size_t num_rows, size_t num_cols){
//	/*
//	Note: will overwrite :result: with the result of the computation.
//	*/
//	cblas_dgemv(CblasRowMajor, CblasNoTrans, num_rows, num_cols, 1, arr, num_cols, vec, 1, 0, result,1);
//}
//
//
//void openblas_dot_matrix_vec_t(double* arr, double* vec, double* result, size_t num_rows, size_t num_cols){
//	/*
//	Note: will overwrite :result: with the result of the computation.
//	*/
//	cblas_dgemv(CblasRowMajor, CblasTrans, num_rows, num_cols, 1, arr, num_cols, vec, 1, 0, result ,1);
//}
//
//
//void openblas_dot_matrix_vec_add(double* arr, double* vec, double* result, size_t num_rows, size_t num_cols){
//	/*
//	Note: will add the result of the computation to the existing contents of :result:
//	*/
//	cblas_dgemv(CblasRowMajor, CblasNoTrans, num_rows, num_cols, 1, arr, num_cols, vec, 1, 1, result,1);
//}
//
//
//void openblas_dot_matrix_vec_add_t(double* arr, double* vec, double* result, size_t num_rows, size_t num_cols){
//	/*
//	Note: will add the result of the computation to the existing contents of :result:
//	*/
//	cblas_dgemv(CblasRowMajor, CblasTrans, num_rows, num_cols, 1, arr, num_cols, vec, 1, 1, result ,1);
//}


void mult_vec_scalar(double* source_vec, double scalar, double* dest_vec, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		dest_vec[element_idx] = source_vec[element_idx] * scalar;
	}
}


void mult_assign_vec_scalar(double* vec, double scalar, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		vec[element_idx] *= scalar;
	}
}


void mult_vec_vec(double* vec_a, double* vec_b, double* dest_vec, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		dest_vec[element_idx] = vec_a[element_idx] * vec_b[element_idx];
	}
}


void mult_assign_vec_vec(double* vec_a, double* vec_b, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		vec_a[element_idx] *= vec_b[element_idx];
	}
}


void div_assign_vec_scalar(double* vec, double scalar, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		vec[element_idx] /= scalar;
	}
}


void div_vec_scalar(double* source_vec, double scalar, double* dest_vec, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		dest_vec[element_idx] = source_vec[element_idx] / scalar;
	}
}


void div_assign_vec_vec(double* vec_a, double* vec_b, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		vec_a[element_idx] /= vec_b[element_idx];
	}
}


void add_vec_scalar(double* source_vec, double scalar, double* dest_vec, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		dest_vec[element_idx] = source_vec[element_idx] * scalar;
	}
}
	
	
void add_assign_vec_scalar(double* vec, double scalar, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		vec[element_idx] += scalar;
	}
}


void add_vec_vec(double* vec_a, double* vec_b, double* dest_vec, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		dest_vec[element_idx] = vec_a[element_idx] * vec_b[element_idx];
	}
}
	
	
void add_assign_vec_vec(double* vec_a, double* vec_b, size_t num_elements){
	size_t element_idx;
	for (element_idx=0; element_idx < num_elements; element_idx++){
		vec_a[element_idx] += vec_b[element_idx];
	}
}


double sum_vec(double* vec, size_t end_pos){
	double accumulator = 0.0;
	size_t element_idx;
	
	for (element_idx = 0; element_idx < end_pos; element_idx++){
		accumulator += vec[element_idx];
	}	
	return accumulator;
}

void print_vec_d(double* vec, size_t num_cols){
    size_t col_idx;
    for (col_idx=0; col_idx < num_cols; col_idx++){
        // printf("%p %f ", &vec[col_idx], vec[col_idx]);
		printf("%f ", vec[col_idx]);

    }
    printf("\n");
}

void print_vec_l(long* vec, size_t num_cols){
    size_t col_idx;
    for (col_idx=0; col_idx < num_cols; col_idx++){
        // printf("%p %zu ", &vec[col_idx], vec[col_idx]);
		printf("%zu ", vec[col_idx]);

    }
    printf("\n");
}


void print_matrix(double** matrix, size_t num_rows, size_t num_cols){
    size_t row_idx;
    
    for (row_idx=0; row_idx < num_rows; row_idx++){
        // printf("Row %zu, first element address: %p\n", row_idx, matrix[row_idx]);
        print_vec_d(matrix[row_idx], num_cols);
    }
}


void fill_vec(double* vec, size_t num_elements, double fill_value){
	size_t element_idx;
	//printf("%f\n",array2d);
	for (element_idx=0;element_idx < num_elements; element_idx++){
		// printf("%f ", vec[element_idx]);
		vec[element_idx] = fill_value;
	}
}

void shuffle_array_d(double* arr, size_t n)
// A function to generate a random permutation of arr[]
{
    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (size_t i = n-1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        size_t j = rand() % (i+1);

        // Swap arr[i] with the element at random index
        double temp = arr[j];
        arr[j] = arr[i];
        arr[i] = temp;
//        swap(&arr[i], &arr[j]);
    }
}

void shuffle_array_st(size_t* arr, size_t n)
// A function to generate a random permutation of arr[]
{
    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (size_t i = n-1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        size_t j = rand() % (i+1);

        // Swap arr[i] with the element at random index
        size_t temp = arr[j];
        arr[j] = arr[i];
        arr[i] = temp;
//        swap(&arr[i], &arr[j]);
    }
}
