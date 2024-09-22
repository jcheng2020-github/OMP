import numpy as np

class OMP:
    # Custom implementation of 2-norm
    def custom_norm(self, v):
        return np.sqrt(np.sum(v**2))

    # Custom implementation of pseudo-inverse using Singular Value Decomposition (SVD)
    def custom_pinv(self, A):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s_inv = np.zeros_like(s)
        for i in range(len(s)):
            if s[i] > 1e-5:  # Avoid division by zero
                s_inv[i] = 1.0 / s[i]
        return Vt.T @ np.diag(s_inv) @ U.T

    # Read the data from txt files
    def read_data(self, file_name):
        return np.loadtxt(file_name)

    # Step 2: Create the feature matrix Φ
    def create_feature_matrix(self, x, dx1, dx2, dx3):
        phi = np.column_stack([
            dx1, dx2, dx3,                      # Linear derivatives
            x * dx1, x * dx2, x * dx3,          # Terms with x multiplied by derivatives
            (x**2) * dx1, (x**2) * dx2, (x**2) * dx3,  # Quadratic terms with x^2
            (x**3) * dx1, (x**3) * dx2, (x**3) * dx3   # Cubic terms with x^3
        ])
        return phi

    # Step 3: Normalize the columns of the feature matrix
    def normalize_columns(self, phi):
        norms = np.array([self.custom_norm(phi[:, j]) for j in range(phi.shape[1])])  # Compute the 2-norm of each column
        return phi / norms, norms

    

    # Step 4: Orthogonal Matching Pursuit (OMP)
    def omp(self, phi, x, tol=1e-2):
        N = len(x)
        r = x.copy()  # Initialize the residual as x
        indices = []  # Set of chosen indices
        a_e = np.zeros(phi.shape[1])  # Coefficient vector (initially zeros)

        while self.custom_norm(r) ** 2 / N > tol:
            # Step 4: Compute y = φ^T * r (inner products to find the "similarities")
            y = np.dot(phi.T, r)

            # Step 5: Identify the index i_opt with the maximum absolute value of y
            i_opt = np.argmax(np.abs(y))
            indices.append(i_opt)

            # Step 7: Solve for the sparse coefficients a_e on the selected columns
            phi_selected = phi[:, indices]
            a_selected = np.dot(self.custom_pinv(phi_selected), x)  # Least squares solution for selected features

            # Step 8: Update the residual r = x - φ_selected * a_selected
            r = x - np.dot(phi_selected, a_selected)

        # Step 9: Compute the final denormalized coefficients a
        a_final = np.zeros(phi.shape[1])
        a_final[indices] = a_selected / norms[indices]  # Denormalize the coefficients
        return a_final

task = OMP()
x = task.read_data('x.txt')  # The x(t) values
dx1 = task.read_data('dx1.txt')  # First derivative data
dx2 = task.read_data('dx2.txt')  # Second derivative data
dx3 = task.read_data('dx3.txt')  # Third derivative data
# Construct the feature matrix
phi = task.create_feature_matrix(x, dx1, dx2, dx3)

phi_normalized, norms = task.normalize_columns(phi)

# Step 5: Run the OMP algorithm to get the sparse coefficients
coefficients = task.omp(phi_normalized, x)

# Output the final coefficients
print("Sparse coefficients: ", coefficients)
