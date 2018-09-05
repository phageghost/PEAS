#include <stdlib.h>
#include <stdio.h>
#include "my_array_funcs.h"

// #define printf PySys_WriteStdout
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

void c_scan_emissions(long* observations,
					  double* emissions,
					  size_t num_states,
					  size_t num_symbols,
					  size_t num_observations,
					  double* scores){

	double **e_rows;
	size_t observation_idx, state_idx;

	e_rows = get_row_ptrs(emissions, num_symbols, num_states);


	for (observation_idx = 0; observation_idx < num_observations - num_states; observation_idx++){
		for (state_idx = 0; state_idx < num_states; state_idx++){
			// printf("state_idx %zu, observation %zu", state_idx, observations[observation_idx + state_idx]);
			scores[observation_idx] *= e_rows[observations[observation_idx + state_idx]][state_idx];
		}
	}
}



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


void c_forward_backward_silent_native_sparse(size_t num_submodels,
										  size_t* model_starts,
										  size_t* model_ends,
										  double* from_silent,
										  double* emissions,
										  double* forward_prior,
										  long* observations,
										  double* posterior_decoding,
										  size_t  num_states,
										  size_t  num_symbols,
										  size_t  num_observations){
    /* Important note about indexing. The forward pass uses the 0th row for the forward prior.
	Therefore the sequence positions begin with 1 and end with len(observations)*/

	double *f, *b, *c_s;
	double *state_buffer;
	double silent_belief;
	double **f_rows, **b_rows, **e_rows;
	const size_t maxsize = 0 - 1;
	size_t observation_idx, state_idx, submodel_idx;

	// Forward meessages. num_observations+1 rows, num_states cols
	f = (double*) calloc(num_states * (num_observations + 1), sizeof(double));
	// Forward meessages. num_observations rows, num_states cols
	b = (double*) calloc(num_states * num_observations, sizeof(double));
	// normalization. num_obserations elements
	c_s = (double*) calloc(num_observations + 1, sizeof(double));
	// Backward prior. num_states elements.
	// backward_prior = (double*) calloc(num_states, sizeof(double));
	state_buffer = (double*) calloc(num_states, sizeof(double));

	f_rows = get_row_ptrs(f, num_observations+1, num_states);
	b_rows = get_row_ptrs(b, num_observations, num_states);
	e_rows = get_row_ptrs(emissions, num_symbols, num_states);

	// Generate forward messages
	f_rows[0] = forward_prior;

	c_s[0] = sum_vec(f_rows[0], num_states);

	// print_matrix(f_rows, num_observations, num_states);
	// printf("From silent: ");
	// print_vec_d(from_silent, num_submodels);


	for (observation_idx=1; observation_idx<num_observations+1; observation_idx++){
		// printf("\nProcessing observation %zu: %i ...\n", observation_idx, observations[observation_idx-1]);

		// First propagate belief from state module endpoints into the silent state
		silent_belief = 0;
		for (submodel_idx=0; submodel_idx < num_submodels; submodel_idx++){
			// printf("submodel %zu, model end %zu, transition weight %f\n",  submodel_idx, model_ends[submodel_idx], from_silent[submodel_idx]);
			silent_belief += f_rows[observation_idx-1][model_ends[submodel_idx]];
		}

		// silent_belief
		// silent_belief = dot_product_vec_vec(to_silent, f_rows[observation_idx-1], num_states);
		// printf("Silent belief %f\n", silent_belief);

		// Now propagate belief through the submodules (simply flows from start to end, one state position per base pair)
		for (submodel_idx=0; submodel_idx < num_submodels; submodel_idx++){
			// printf("Submodel %zu\n", submodel_idx);
			for (state_idx=model_starts[submodel_idx]; state_idx<model_ends[submodel_idx]; state_idx++){
				// printf("Past state %zu, copying %f to state %zu\n", state_idx, f_rows[observation_idx-1][state_idx], state_idx+1);
				f_rows[observation_idx][state_idx+1] = f_rows[observation_idx-1][state_idx];
			}
		}
		// openblas_dot_matrix_vec(transitions, f_rows[observation_idx-1], f_rows[observation_idx], num_states, num_states);

		// printf("Transitioned non-silent\n");
		// print_vec_d(f_rows[observation_idx], num_states);

		// Now flow from silent into the entry points of the state modules.
		for (submodel_idx=0; submodel_idx < num_submodels; submodel_idx++){
			// printf("Adding %f X %f to first position %zu of submodel %zu\n", silent_belief, from_silent[submodel_idx], model_starts[submodel_idx], submodel_idx);
			f_rows[observation_idx][model_starts[submodel_idx]] += silent_belief * from_silent[submodel_idx];
		}

		// printf("Received from silent\n");
		// print_vec_d(f_rows[observation_idx], num_states);


		// Condition on emission probabilities
		mult_assign_vec_vec(f_rows[observation_idx], e_rows[observations[observation_idx-1]], num_states);
		// printf("Emission probabilities:\n");
		// print_vec_d(e_rows[observations[observation_idx-1]], num_states);
		// printf("Conditioned on emissions\n");
		// print_vec_d(f_rows[observation_idx], num_states);

		// Compute normalization factor
		c_s[observation_idx] = sum_vec(f_rows[observation_idx], num_states);

		// Perform normalization
		div_assign_vec_scalar(f_rows[observation_idx], c_s[observation_idx], num_states);

		// printf("Normalized by %f\n", c_s[observation_idx]);
		// print_vec_d(f_rows[observation_idx], num_states);

	}
	// printf("Done with forward.\n");
	// print_matrix(f_rows, num_observations+1, num_states);

	// printf("\nGenerating backward messages...");
	// Generate backward messages
	fill_vec(b_rows[num_observations-1], num_states, 1);

	// print_matrix(b_rows, num_observations, num_states);
	for (observation_idx=num_observations - 2; observation_idx != maxsize; observation_idx--){
		// printf("\nProcessing observation %zu: %i ...\n", observation_idx, observations[observation_idx]);
		// Condition on future emission
		mult_vec_vec(b_rows[observation_idx+1], e_rows[observations[observation_idx+1]], state_buffer, num_states);

		// printf("Conditioned on future emission\n");
		// print_vec_d(state_buffer, num_states);

		// Propagate back into silent
		silent_belief = 0;
		for (submodel_idx=0; submodel_idx < num_submodels; submodel_idx++){
			silent_belief += state_buffer[model_starts[submodel_idx]] * from_silent[submodel_idx];
		}

		// silent_belief = dot_product_vec_vec(state_buffer, from_silent, num_states);
		// printf("Silent belief %f\n", silent_belief);


		// Propagate non-silent states
		for (submodel_idx=0; submodel_idx < num_submodels; submodel_idx++){
			for (state_idx=model_starts[submodel_idx]; state_idx<model_ends[submodel_idx]; state_idx++){
				b_rows[observation_idx][state_idx] = state_buffer[state_idx+1];
			}
		}

		// openblas_dot_matrix_vec_t(transitions, state_buffer, b_rows[observation_idx], num_states, num_states);
		// printf("Non-silent propagation\n");
		// print_vec_d(state_buffer, num_states);
		// print_vec_d(b_rows[observation_idx], num_states);


		// Propagate back from silent
		for (submodel_idx=0; submodel_idx < num_submodels; submodel_idx++){
			b_rows[observation_idx][model_ends[submodel_idx]] += silent_belief;
		}

		// for (state_idx=0; state_idx<num_states; state_idx++){
			// b_rows[observation_idx][state_idx] += to_silent[state_idx] * silent_belief;
		// }
		// printf("Back from silent\n");
		// print_vec_d(b_rows[observation_idx], num_states);


		// Normalize
		div_assign_vec_scalar(b_rows[observation_idx], c_s[observation_idx+2], num_states);

		// printf("Normalized by %f\n", c_s[observation_idx+1]);
		// print_vec_d(b_rows[observation_idx], num_states);

	}
	// printf("Done with backward.\n");

	// printf("\nF:\n");
	// print_matrix(f_rows, num_observations+1, num_states);
	// printf("\nB:\n");
	// print_matrix(b_rows, num_observations, num_states);


	// Multiply f (all but last row) and b to get the posterior decoding
	for (observation_idx=0; observation_idx<num_observations; observation_idx++){
		for (state_idx=0; state_idx<num_states; state_idx++){
			// printf("Obs %zu State %zu\n", observation_idx, state_idx);
			// printf("Multiplying %f x %f = %f\n", f_rows[observation_idx][state_idx], b_rows[observation_idx][state_idx], f_rows[observation_idx][state_idx] * b_rows[observation_idx][state_idx]);
			posterior_decoding[observation_idx * num_states + state_idx] = f_rows[observation_idx+1][state_idx] * b_rows[observation_idx][state_idx];
			// printf("assigned %f\n", posterior_decoding[observation_idx * num_states + state_idx]);
		}
	}

		// Time to take out the trash
	// for (observation_idx=0; observation_idx<num_obserations+1; observations++){
		// free(f_rows[observation_idx]);
	// }

	// for (observation_idx=0; observation_idx<num_obserations; observations++){
		// free(b_rows[observation_idx]);
	// }
	free(f_rows);
	free(b_rows);
	free(e_rows);
	free(f);
	free(b);
	free(c_s);
	free(state_buffer);

    return;
}


