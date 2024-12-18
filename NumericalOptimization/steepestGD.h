//
// Created by jiance on 24-12-18.
//

#ifndef STEEPESTGD_H
#define STEEPESTGD_H

#include <iostream>
#include <cmath>
#include <vector>

class steepestGD {
public:
    double rosenbrock(std::vector<double> &x);
    std::vector<double> rosenbrock_grad(std::vector<double> &x);
    std::vector<double> steepest_descent(double alpha, double epsilon, int max_iter);
private:

};

double steepestGD::rosenbrock(std::vector<double> &x) {
    return (1 - x[0]) * (1 - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
}

std::vector<double> steepestGD::rosenbrock_grad(std::vector<double> &x) {
    std::vector<double> gradient(2);
    gradient[0] = -2 * (1 - x[0]) - 4 * 100 * x[0] * (x[1] - x[0] * x[0]);
    gradient[1] = 2 * 100 * (x[1] - x[0] * x[0]);
    return gradient;
}

std::vector<double> steepestGD::steepest_descent(double alpha, double epsilon, int max_iter) {
    std::vector<double> x_0 = {-1.2, 1};
    for (int iter = 0; iter < max_iter; iter++) {
        auto grad = rosenbrock_grad(x_0);
        double grad_norm = std::sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

        if (grad_norm < epsilon) {
            return x_0;
        }

        x_0[0] -= alpha*grad[0];
        x_0[1] -= alpha*grad[1];
    }

    return x_0;
}

#endif //STEEPESTGD_H
