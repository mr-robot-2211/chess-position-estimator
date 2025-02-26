import numpy as np

# Function to read a matrix from a .txt file
def read_matrix_from_file(file_path):
    matrix = []
    max_length = 0
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()
            max_length = max(max_length, len(row))  # Track the longest row
            matrix.append(row)
    
    # Pad shorter rows to match the longest row
    for i in range(len(matrix)):
        matrix[i].extend(['x'] * (max_length - len(matrix[i])))  # Add 'x' padding to shorter rows
    
    return matrix

# Function to pad smaller matrix with 'x' in the beginning to match the row count of the larger matrix
def pad_matrix(matrix, target_row_count):
    current_row_count = len(matrix)
    # Add rows at the start of the matrix until the row count matches the target
    for _ in range(target_row_count - current_row_count):
        matrix.insert(0, ['x'] * len(matrix[0]))  # Pad with 'x' for each column in the row
    return matrix

# Read matrices from the two .txt files
matrix2 = read_matrix_from_file('/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/pieces.txt')
matrix1 = read_matrix_from_file('/Users/yaswanthbitspilani/Documents/GitHub/SOP-3-1/pieces2.txt')

# Pad the smaller matrix to match the row count of the larger matrix
if len(matrix1) > len(matrix2):
    matrix2 = pad_matrix(matrix2, len(matrix1))
elif len(matrix2) > len(matrix1):
    matrix1 = pad_matrix(matrix1, len(matrix2))

# Convert the matrices to numpy arrays for easier manipulation
matrix1 = np.array(matrix1)
matrix2 = np.array(matrix2)

# Combine the matrices based on the logic
for i in range(matrix1.shape[0]):  # Iterate through rows
    for j in range(matrix1.shape[1]):  # Iterate through columns
        if matrix1[i, j] == 'x' or matrix2[i,j] == 'x':  # If matrix1 has 'x', keep the content from matrix2
            continue
        elif matrix1[i, j] in ['0', '1']:  # If matrix1 has 0 or 1, overwrite matrix2 with the same value
            matrix2[i, j] = matrix1[i, j]

# Print the resulting combined matrix
print("Combined Matrix:")
print(matrix2)
