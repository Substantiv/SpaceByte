#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

// 畸变后的图像
string image_file = "./test.png";

int main(int argc, char **argv) {

    // 畸变参数
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // 内参
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    cv::Mat image = cv::imread(image_file,0);   // 图像是灰度图，CV_8UC1
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);   // 去畸变以后的图

    // 计算去畸变后图像的内容
    for (int v = 0; v < rows; v++)
        for (int u = 0; u < cols; u++) {

            // 像素坐标系转换为归一化坐标系
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;

            // 计算归一化后的平面
            r_2 = pow(x, 2) + pow(y, 2);
            double x_distorted = x*(1+k1*r_2+k2*pow(r_2,2)) + 2*p1*x*y + p2*(r_2+2*x*x);
            double y_distorted = y*(1+k1*r_2+k2*pow(r_2,2)) + p1*(pow(r_2,2)+2*y*y) + 2*p2*x*y;

            // 归一化坐标系转换为像素坐标系
            u_distorted = x*fx + cx;
            v_distorted = y*fy + cy;

            // 赋值 (最近邻插值)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            } else {
                image_undistort.at<uchar>(v, u) = 0;
            }
        }

    // 画图去畸变后图像
    cv::imshow("image undistorted", image_undistort);
    cv::waitKey();

    return 0;
}
