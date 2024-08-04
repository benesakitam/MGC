import os
import json
import shutil


def mkdir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def generate_dataset(dataset_path, cls_imgs):
    dataset_train_path = os.path.join(dataset_path, 'train')
    mkdir(dataset_train_path)
    dataset_val_path = os.path.join(dataset_path, 'val')
    mkdir(dataset_val_path)

    for cls, imgs in cls_imgs.items():
        train_folder = os.path.join(dataset_train_path, cls)
        mkdir(train_folder)

        for img in imgs:
            original_path = os.path.join(dataset_path, cls, img)
            new_path = os.path.join(train_folder, img)
            shutil.move(original_path, new_path)

        remain_imgs_folder = os.path.join(dataset_path, cls)
        shutil.move(remain_imgs_folder, dataset_val_path)


with open('./datasets_selected_img_names.json', 'r') as f:
    datasets_img_names = json.load(f)

# each path of your scene classification datasets
datasets_path = ['/workspace/Dataset/AID/',
                 '/workspace/Dataset/MLRSNet/',
                 '/workspace/Dataset/NWPU45/']

for i, (_, cls_imgs) in enumerate(datasets_img_names.items()):
    generate_dataset(datasets_path[i], cls_imgs)

