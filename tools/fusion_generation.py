import os
import json
import shutil

with open('./datasets_selected_img_names.json', 'r') as f:
    datasets_selected_img_names = json.load(f)

# the path of Fusion dataset
Fusion_path = "/workspace/Dataset/Fusion/"
if not os.path.exists(Fusion_path):
    os.mkdir(Fusion_path)

# the path of training set in Fusion dataset
Fusion_train_path = "/workspace/Dataset/Fusion/train/"
if not os.path.exists(Fusion_train_path):
    os.mkdir(Fusion_train_path)

# the unique path of your datasets
original_path = "/workspace/Dataset/"

cls_map = {
    'baseballfield': 'baseball_diamond',
    'commercial': 'commercial_area',
    'denseresidential': 'dense_residential_area',
    'industrial': 'industrial_area',
    'mediumresidential': 'medium_residential',
    'parking': 'parking_lot',
    'playground': 'ground_track_field',
    'pond': 'lake',
    'port': 'harbor&port',
    'railwaystation': 'railway_station',
    'sparseresidential': 'sparse_residential_area',
    'storagetanks': 'storage_tank',
    'dense_residential': 'dense_residential_area',
    'harbor': 'harbor&port',
    'sparse_residential': 'sparse_residential_area'
}

fusion_all_cls_names = {}

for dataset, cls_imgs in datasets_selected_img_names.items():
    for cls, imgs in cls_imgs.items():
        if cls.lower() in cls_map:
            fusion_cls_name = cls_map[cls.lower()]
        else:
            fusion_cls_name = cls.lower()

        fusion_cls_path = os.path.join(Fusion_train_path, fusion_cls_name)
        if not os.path.exists(fusion_cls_path):
            os.mkdir(fusion_cls_path)

        fusion_all_cls_names[cls] = fusion_cls_path

        for img in imgs:
            per_dataset_img_path = os.path.join(original_path, dataset, cls, img)
            fusion_img_path = os.path.join(fusion_all_cls_names[cls], str(dataset).lower() + '_' + img)
            shutil.copy(per_dataset_img_path, fusion_img_path)

