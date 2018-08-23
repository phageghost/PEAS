## ROPE: Region of proximal enrichment

# Modules:
	1. Different front-ends for different types of data. Currently just implement one for peaks (include BED and HOMER format).
		1. Split up tag counts and annotations.
		2. Pre-process tag counts, normalize, etc.
		3. Divide into array of chromosome tables (sample by locus).
		4. Calls modules 2 and 3
		4. Post-processing:
			* Generates combined data frame with annotation information previously split off, and compound "peak names"

	2. Wrapper that calls 3 and 4 for a single:
		* Vector
		* 2D matrix
			
	3. Scoring and distributions:
		* For a given 1D-vector
			* 1D case:
				1. Apply some scoring function (implemented ourselves DP-wise, no need to allow arbitrary functions since that will screw up runtime):
					1. Sum
					2. Mean
					3. Max
					4. Min
					5. Median? Trickier to implement, skip for now.
				2. Generate empirical distribution array by region size
			* 2D case:
				1. Apply some scoring function (implemented ourselves DP-wise, no need to allow arbitrary functions since that will screw up runtime):
					1. Sum
					2. Mean
					3. Max
					4. Min
					5. Median? Trickier to implement, skip for now.
				2. Generate approximated empirical distribution array by region size.
			

	4. Perform the dynamic programming algorithm for each chromosome or other data group.
		* As input this takes:
			1. A square matrix of score values (either mean of square matrix (2D) or mean of diagonal region (1D).
			2. An array of distributions by region size
		* As output, gives region coordinates, scores, and p-values.
		

		
		
	5. Plotting functions:
	
	6. Command line scripts
	    1. For peak data (BED files)
	    2. For a single CSV vector or matrix (command line flag for 1D or 2D).
		