from samples.coco.generator import KerasGenerator
from tqdm import tqdm

gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_val2017.json',
                     dataset_dir='coco_dataset',
                     subset='val',
                     year='2017',
                     batch_size=2)
gen.prepare()

gen_try = gen.generate_batch()

for i in tqdm(range(10)):
    next(gen_try)
