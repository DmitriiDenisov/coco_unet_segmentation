from samples.coco.generator import KerasGenerator

gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_val2017.json',
                     dataset_dir='coco_dataset',
                     subset='val',
                     year='2017',
                     batch_size=2)
gen.prepare()

gen_try = gen.generate_batch()
a, b = next(gen_try)
c, d = next(gen_try)
next(gen_try)
next(gen_try)
