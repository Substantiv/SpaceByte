# 文件结构

```
├── CMakeLists.txt -> /opt/ros/melodic/share/catkin/cmake/toplevel.cmake
├── README.md
├── mbot_description            # mbot的urdf模型和xacro模型显示
│   ├── CMakeLists.txt
│   ├── config
│   │   ├── mbot_urdf.rviz
│   │   └── mbot_xacro.rviz
│   ├── doc
│   ├── launch
│   │   ├── display_mbot_urdf.launch
│   │   └── display_mbot_xacro.launch
│   ├── meshes
│   │   └── kinect.dae
│   ├── package.xml
│   └── urdf
│       ├── mbot_base.urdf
│       ├── mbot_base.xacro
│       └── sensor
│           ├── camera.xacro
│           ├── kinect.xacro
│           └── laser.xacro
├── mbot_gazebo
│   ├── CMakeLists.txt
│   ├── config
│   │   └── mbot_gazebo.rviz
│   ├── doc
│   │   ├── mbot.gv
│   │   └── mbot.pdf
│   ├── include
│   │   └── mbot_gazebo
│   ├── launch
│   │   ├── mbot_teleop.launch
│   │   └── view_mbot_gazebo.launch
│   ├── meshes
│   │   └── kinect.dae
│   ├── package.xml
│   ├── scripts
│   │   └── mbot_teleop.py
│   ├── src
│   │   └── mpc_node.cpp
│   ├── urdf
│   │   ├── mbot_base.xacro
│   │   └── sensor
│   │       ├── camera.xacro
│   │       ├── kinect.xacro
│   │       └── laser.xacro
│   └── worlds
│       ├── box_house.world
│       └── hometown_room.world
└── nav_demo
    ├── CMakeLists.txt
    ├── config
    │   ├── map_nav.rviz
    │   ├── mbot_amcl.rviz
    │   └── mbot_move_base.rviz
    ├── launch
    │   ├── mbot_amcl.launch
    │   ├── nav_amcl.launch
    │   ├── nav_map_saver.launch
    │   ├── nav_map_server.launch
    │   ├── nav_mbot.launch
    │   ├── nav_path.launch
    │   ├── nav_sim.launch
    │   └── nav_slam.launch
    ├── map
    │   ├── nav.pgm
    │   └── nav.yaml
    ├── package.xml
    └── param
        ├── base_local_planner_params.yaml
        ├── costmap_common_params.yaml
        ├── global_costmap_params.yaml
        └── local_costmap_params.yaml
```

# 添加功能

* 增加Timer定时器, 接受定位信息进而设计控制器
[ROS中定时器的基本使用](https://blog.csdn.net/qq_45950023/article/details/127496321)

* 增加CasADi求解QP问题[非线性求解器Casadi](https://blog.csdn.net/qq_35632833/article/details/124507599)


# 查漏补缺

* 回调函数使用常量引用类型作为形参的目的是什么

常量引用类型是为了高效地传递对象，同时确保在回调函数中不会修改它。使用常量引用避免了不必要的复制开销，特别是当事件对象较大时。

const 修饰表示常量; & 修饰表示引用

* C++ 引用类型

C++ 中的引用是一种类型，用于创建一个变量的别名。引用允许你通过新的名字来访问已有的变量，而不是创建该变量的副本。一旦把引用初始化为某个变量，就可以使用该引用名称或变量名称来指向变量。

* C++ 引用 vs 指针

引用很容易与指针混淆，它们之间有三个主要的不同：

    - 不存在空引用。引用必须连接到一块合法的内存。
    - 一旦引用被初始化为一个对象，就不能被指向到另一个对象。指针可以在任何时候指向到另一个对象。
    - 引用必须在创建时被初始化。指针可以在任何时间被初始化。
