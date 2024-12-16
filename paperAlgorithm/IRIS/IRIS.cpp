//
// Created by jiance on 24-12-16.
//

#include "IRIS.h"

void IRIS::compute() {
    Eigen::Vector2d center = startPoint_;
    // The default shape of the ellipse is a unit circle (symmetric and unscaled), and it is gradually adjusted
    // to the shape of the largest feasible region as the algorithm iterates.
    // Q is a symmetric positive definite matrix: the diagonal elements Q11 and Q22 represent the lengths of the two axes,
    // while the off-diagonal elements Q12 and Q21 represent the rotation of the ellipse.
    Eigen::MatrixXd ellipse = Eigen::MatrixXd::Identity(2, 2);

    for (int iter = 0; iter < maxIterations_; ++iter) {
        std::cout << "Iteration: " << iter + 1 << std::endl;

        Eigen::MatrixXd separatingPlanes = computeSeparatingPlanes(center, ellipse);
        Eigen::MatrixXd newEllipse = solveSdp(separatingPlanes);
        Eigen::Vector2d newCenter = computeNewCenter(newEllipse);

        if ((newCenter - center).norm() < tolerance_) {
            std::cout << "Converged in " << iter + 1 << " iterations.\n";
            break;
        }

        center = newCenter;
        ellipse = newEllipse;
    }

    center_ = center;
    ellipse_ = ellipse;
}

Eigen::MatrixXd IRIS::computeSeparatingPlanes(const Eigen::Vector2d& center, const Eigen::MatrixXd& ellipse) {
    Eigen::MatrixXd planes(obstacles_.rows(), 3); // Each row represents a plane: [a, b, c] corresponding to ax + by + c = 0

    // Loop through all obstacles to compute separating planes
    for (int i = 0; i < obstacles_.rows(); ++i) {
        Eigen::Vector2d obstacle = obstacles_.row(i).transpose(); // Extract the obstacle position
        Eigen::Vector2d diff = obstacle - center; // Compute the difference vector from the center to the obstacle
        Eigen::Vector2d normal = diff.normalized(); // Normalize the difference vector to get the normal vector of the plane

        // Plane equation: n · x + c = 0
        double offset = -normal.dot(obstacle); // Compute the offset 'c' for the plane equation

        // Set the coefficients of the plane equation: normal vector components and the offset
        planes(i, 0) = normal(0); // Coefficient a of the plane (normal vector x-component)
        planes(i, 1) = normal(1); // Coefficient b of the plane (normal vector y-component)
        planes(i, 2) = offset;    // Coefficient c of the plane (offset)

    }

    return planes; // Return the matrix of separating planes
}

Eigen::MatrixXd IRIS::solveSdp(const Eigen::MatrixXd& separatingPlanes) {
    int n = 2;  // Dimension of the ellipse (2D ellipse)
    int m = separatingPlanes.rows();  // Number of separating planes

    // Objective: Minimize trace(Q) --> min tr(Q * P)
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n*n, n*n);  // Create a zero matrix for P
    for (int i = 0; i < n; ++i) {
        P(i*n+i, i*n+i) = 1.0;  // Set diagonal elements of P to 1.0
    }

    // Linear constraints: Ai * Q <= bi
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n*n);  // Zero vector for the linear term in the objective function
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n*n);  // Matrix for the linear constraints
    Eigen::VectorXd l = Eigen::VectorXd::Zero(m);  // Lower bounds for the constraints
    Eigen::VectorXd u = Eigen::VectorXd::Zero(m);  // Upper bounds for the constraints

    // Loop through each separating plane to populate A, l, and u
    for (int i = 0; i < m; ++i) {
        Eigen::Vector2d normal = separatingPlanes.row(i).segment(0, 2);  // Extract normal vector (2D)
        double offset = separatingPlanes(i, 2);  // Extract offset for the plane

        // Create the matrix Ni (outer product of the normal vector)
        Eigen::MatrixXd Ni = normal * normal.transpose();

        // Populate the constraint matrix A
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                A(i, j * n + k) = Ni(j, k);  // Set A to the elements of Ni
            }
        }

        // Set the bounds for the constraint: Ai * Q <= bi
        l(i) = -std::numeric_limits<double>::infinity();  // No lower bound
        u(i) = -offset;  // Upper bound based on the offset of the separating plane
    }

    // Initialize the OSQP solver
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(true);  // Set verbosity for debugging
    solver.settings()->setWarmStart(true);  // Enable warm-starting

    // Set the dimensions of the problem for the solver
    solver.data()->setNumberOfVariables(n * n);  // Number of variables (Q is n*n)
    solver.data()->setNumberOfConstraints(m);   // Number of constraints (m separating planes)

    // Set the problem matrices for OSQP solver
    solver.data()->setHessianMatrix(_convertToSparseMatrix(P));  // Set P matrix (converted to sparse)
    solver.data()->setGradient(q);  // Set gradient (q), which is zero in this case
    solver.data()->setLinearConstraintsMatrix(_convertToSparseMatrix(A));  // Set A matrix (constraints)
    solver.data()->setLowerBound(l);  // Set lower bounds for the constraints
    solver.data()->setUpperBound(u);  // Set upper bounds for the constraints

    // Initialize the solver and solve the problem
    solver.initSolver();
    solver.solveProblem();

    // Extract the solution vector (flattened matrix Q)
    Eigen::VectorXd solution = solver.getSolution();

    // Convert the solution vector into a matrix form (ellipse parameters)
    Eigen::MatrixXd newEllipse = Eigen::MatrixXd::Zero(n, n);  // Initialize the new ellipse matrix
    Eigen::VectorXd Q_vec = Eigen::Map<Eigen::VectorXd>(solution.data(), n * n);  // Map the solution into a vector

    // Fill the new ellipse matrix with the values from the solution vector
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            newEllipse(i, j) = Q_vec(i * n + j);  // Populate the matrix
        }
    }

    // Return the new ellipse matrix
    return newEllipse;
}

Eigen::Vector2d IRIS::computeNewCenter(const Eigen::MatrixXd& ellipse) {
    return Eigen::Vector2d::Zero();
}

Eigen::SparseMatrix<double> IRIS::_convertToSparseMatrix(Eigen::MatrixXd A)
{
    int row = A.rows();
    int col = A.cols();
    Eigen::SparseMatrix<double> A_s(row, col);

    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            A_s.insert(i, j) = A(i, j);

    return A_s;
}