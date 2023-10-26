import csv
import os
import shutil

def organize_images(data_dir, labels_file, start_idx, end_idx):
    # 從 CSV 文件中讀取標籤
    with open(labels_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳過標頭行
        for i, row in enumerate(reader):
            if i < start_idx or i >= end_idx:
                continue
            image_file, label = row[0] + '.png', row[1]
            # 為每個類別創建目錄
            label_dir = os.path.join(data_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            # 將圖片移動到相應的目錄
            shutil.move(os.path.join(data_dir, image_file), os.path.join(label_dir, image_file))

# 指定路徑和範圍
labels_file = './data/CIFAR10/trainLabels.csv'
validation_dir = './data/CIFAR10/validation'
validation_range = (45000, 50000)

organize_images(validation_dir, labels_file, *validation_range)
