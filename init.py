import os

from Download_utils import download_file_tqdm

os.system("mkdir coco_dataset_1")
os.system("mkdir coco_dataset_1/images")

download_file_tqdm('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', 'annotations_2017.zip')
os.system("unzip annotations_2017.zip")
os.system("rm -r annotations/captions_train2017.json")
os.system("rm -r annotations/captions_val2017.json")
os.system("rm -r annotations/person_keypoints_train2017.json")
os.system("rm -r annotations/person_keypoints_val2017.json")
os.system("mv annotations coco_dataset_1")

download_file_tqdm('http://images.cocodataset.org/zips/val2017.zip', 'val2017.zip')
os.system("unzip val2017.zip")
os.system("mv val2017/ coco_dataset_1/images/")

download_file_tqdm('http://images.cocodataset.org/zips/train2017.zip', 'train2017.zip')
os.system("unzip train2017.zip")
os.system("mv train2017/ coco_dataset_1/images/")

os.system("rm -r annotations_2017.zip")
os.system("rm -r __pycache__")