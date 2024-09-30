#!/usr/bin/env python
# -*-coding:utf-8-*
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np

class FigureEightPublisher:
    def __init__(self):
        self.publisher = rospy.Publisher('/mpc/traj_point', Float32MultiArray, queue_size=10)
        self.rate = rospy.Rate(0.5)  # 10 Hz
        self.num_points = 100  # 轨迹点数量

    def generate_figure_eight(self):
        t = np.linspace(0, 2 * np.pi, self.num_points)
        x = np.sin(t)  # x 轴
        y = np.sin(2 * t) / 2  # y 轴
        yaw = np.arctan2(np.gradient(y), np.gradient(x))  # 计算偏航角

        return np.vstack((x, y, yaw)).flatten()

    def publish_traj_points(self):
        while not rospy.is_shutdown():
            msg = Float32MultiArray()
            points = self.generate_figure_eight()
            msg.data = points.tolist()
            self.publisher.publish(msg)
            rospy.loginfo("Published trajectory points: %s", msg.data)
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('figure_eight_publisher', anonymous=True)
    publisher = FigureEightPublisher()
    try:
        publisher.publish_traj_points()
    except rospy.ROSInterruptException:
        pass
