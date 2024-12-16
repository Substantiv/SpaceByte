//
// Created by jiance on 24-12-16.
//

#ifndef IRIS_H
#define IRIS_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include <osqp.h>
#include <OsqpEigen/OsqpEigen.h>

class IRIS {
public:
    IRIS(const Eigen::MatrixXd& obstacles, const Eigen::Vector2d& startPoint, double tolerance = 1e-3, int maxIterations = 100)
        : obstacles_(obstacles), startPoint_(startPoint), tolerance_(tolerance), maxIterations_(maxIterations) {}

    Eigen::Vector2d getCenter() const { return center_; }
    Eigen::MatrixXd getEllipse() const { return ellipse_; }

    void compute();

private:
    double tolerance_;
    int maxIterations_;
    Eigen::Vector2d center_;
    Eigen::MatrixXd ellipse_;
    Eigen::MatrixXd obstacles_;
    Eigen::Vector2d startPoint_;

    Eigen::SparseMatrix<double> _convertToSparseMatrix(Eigen::MatrixXd A);

    // Calculation of separating hyperplanes (linear programming)
    Eigen::MatrixXd computeSeparatingPlanes(const Eigen::Vector2d& center, const Eigen::MatrixXd& ellipse);

    // Ellipse region update (semidefinite programming)
    Eigen::MatrixXd solveSdp(const Eigen::MatrixXd& separatingPlanes);

    // Compute the new center based on the ellipse matrix
    Eigen::Vector2d computeNewCenter(const Eigen::MatrixXd& ellipse);

};

#endif //IRIS_H
