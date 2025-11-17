import torch

def rowswap(matrix, source_row, target_row):
    
    # Create a copy to avoid modifying original matrix
    new_matrix = matrix.clone()
    
    # Swap rows
    new_matrix[[source_row, target_row]] = new_matrix[[target_row, source_row]]
    
    return new_matrix


def rowscale(matrix, row_index, scaling_factor):

    # Create a copy to avoid modifying original matrix
    new_matrix = matrix.clone()
    
    # Scale the row
    new_matrix[row_index] = new_matrix[row_index] * scaling_factor
    
    return new_matrix


def rowreplacement(matrix, first_row, second_row, j, k):

    # Create a copy to avoid modifying original matrix
    new_matrix = matrix.clone()
    
    # Perform row replacement: j*R_first + k*R_second
    new_row = j * new_matrix[first_row] + k * new_matrix[second_row]
    new_matrix[second_row] = new_row
    
    return new_matrix


def rref(matrix, tolerance=1e-10):

    # Create a copy and convert to float for numerical stability
    rref_matrix = matrix.clone().float()
    num_rows, num_cols = rref_matrix.shape
    
    lead = 0  # lead column
    
    for r in range(num_rows):
        if lead >= num_cols:
            return rref_matrix
        
        # Find pivot row
        i = r
        while abs(rref_matrix[i, lead]) < tolerance:
            i += 1
            if i == num_rows:
                i = r
                lead += 1
                if num_cols == lead:
                    return rref_matrix
        
        # Swap rows if necessary
        if i != r:
            rref_matrix = rowswap(rref_matrix, i, r)
        
        # Get the pivot value
        pivot = rref_matrix[r, lead]
        
        # Scale pivot row to make pivot element 1
        if abs(pivot) > tolerance:
            rref_matrix = rowscale(rref_matrix, r, 1.0 / pivot)
        
        # Eliminate other rows
        for i in range(num_rows):
            if i != r:
                factor = -rref_matrix[i, lead]
                rref_matrix = rowreplacement(rref_matrix, r, i, factor, 1)
        
        lead += 1
    
    return rref_matrix


def main():

    print("Testing Elementary Row Operations")
    print("=" * 50)
    
    # Test matrix from the problem
    test_matrix = torch.tensor([
        [1, 3, 0, 0, 3],
        [0, 0, 1, 0, 9],
        [0, 0, 0, 1, 4]
    ], dtype=torch.float32)
    
    print("Original Matrix:")
    print(test_matrix)
    print()
    
    # 1. Perform R1 <-> R2 using rowswap
    matrix_after_swap = rowswap(test_matrix, 0, 1)
    print("After swapping R1 and R2:")
    print(matrix_after_swap)
    print()
    
    # 2. Perform (1/3)R1 using rowscale
    matrix_after_scale = rowscale(matrix_after_swap, 0, 1/3)
    print("After scaling R1 by 1/3:")
    print(matrix_after_scale)
    print()
    
    # 3. Perform R3 = 3*R1 + R3 using rowreplacement
    matrix_after_replacement = rowreplacement(matrix_after_scale, 0, 2, 3, 1)
    print("After R3 = 3*R1 + R3:")
    print(matrix_after_replacement)
    print()
    
    # Test RREF on a different matrix
    print("Testing RREF Function")
    print("=" * 50)
    
    # Create a test matrix for RREF
    rref_test_matrix = torch.tensor([
        [2, 1, -1, 8],
        [-3, -1, 2, -11],
        [-2, 1, 2, -3]
    ], dtype=torch.float32)
    
    print("Matrix for RREF test:")
    print(rref_test_matrix)
    print()
    
    rref_result = rref(rref_test_matrix)
    print("RREF result:")
    print(rref_result)
    print()
    
    # Verify the test matrix from problem statement
    print("RREF of original test matrix:")
    original_rref = rref(test_matrix)
    print(original_rref)


if __name__ == "__main__":
    main()
