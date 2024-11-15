/*
* 根据表面法线、曲率变化去除噪声点
  学习如何使用统计分析技术从点云数据集中去除噪声测量值，例如离群值
*/

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
 
int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
 
  // Fill in the cloud data
  pcl::PCDReader reader;
  // Replace the path below with the path where you saved your file
  reader.read<pcl::PointXYZ> ("table_scene_lms400.pcd", *cloud);
 
  std::cerr << "Cloud before filtering: " << std::endl;
  std::cerr << *cloud << std::endl;
 
  // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  // 设置在滤波时计算每个点邻域的点数为 50
  sor.setMeanK (50);
  // 设置一个阈值，如果某点与邻域内的点的平均距离与标准差的倍数超过这个值，
  // 就认为该点是噪声点，进行去除。
  sor.setStddevMulThresh (1.0);
  sor.filter (*cloud_filtered);
 
  std::cerr << "Cloud after filtering: " << std::endl;
  std::cerr << *cloud_filtered << std::endl;
 
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ> ("table_scene_lms400_inliers.pcd", *cloud_filtered, false);
 
  sor.setNegative (true);
  sor.filter (*cloud_filtered);
  writer.write<pcl::PointXYZ> ("table_scene_lms400_outliers.pcd", *cloud_filtered, false);
 
  return (0);
}

