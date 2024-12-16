#include <iostream>
#include "IRIS.h"

int main() {
    // 定义障碍物 (每一行是一个障碍物的中心点)
    Eigen::MatrixXd obstacles(3, 2);
    obstacles << 1.0, 1.0,
                 2.0, 2.0,
                 3.0, 1.0;

    // 起点
    Eigen::Vector2d startPoint(0.0, 0.0);

    // 创建 IRIS 对象
    IRIS iris(obstacles, startPoint);

    // 计算最大椭圆区域
    iris.compute();

    // 输出结果
    std::cout << "最大椭圆中心: \n" << iris.getCenter() << "\n";
    std::cout << "最大椭圆矩阵: \n" << iris.getEllipse() << "\n";

    return 0;
}
