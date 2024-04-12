"""
    根据 image 文件名称创建训练数据的 label 文件
"""

import os

# 路径名称
root_dir = "dataset/train"
target_dir = "bees_image"
img_path = os.listdir(os.path.join(root_dir, target_dir))

# 根据文件名称提取label值
label = target_dir.split('_')[0]
out_dir = 'bees_label'

# 设置label值
for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label)
