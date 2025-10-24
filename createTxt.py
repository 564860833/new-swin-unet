import os

# 定义数据目录路径
train_dir = './data/Synapse/train_npz'
val_dir = './data/Synapse/test_vol_h5'

# 定义输出的列表文件路径
output_dir = './lists/Synapse'
os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建

# 生成 train.txt 文件
train_files = os.listdir(train_dir)
with open(os.path.join(output_dir, 'train.txt'), 'w') as train_f:
    for file_name in train_files:
        if file_name.endswith('.npz'):
            train_f.write(f"{file_name}\n")

# 生成 val.txt 文件
val_files = os.listdir(val_dir)
with open(os.path.join(output_dir, 'val.txt'), 'w') as val_f:
    for file_name in val_files:
        if file_name.endswith('.npy.h5'):
            val_f.write(f"{file_name}\n")

print("train.txt 和 val.txt 文件已成功生成！")
