#include "steepestGD.h"
// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main() {

    std::vector<double> result(2);
    steepestGD steepestGD;
    result = steepestGD.steepest_descent(0.005, 1e-6, 10000);
    std::cout << "Rosenbrock 函数的最小值点为: (" << result[0] << ", " << result[1] << ")\n";
    return 0;
}
