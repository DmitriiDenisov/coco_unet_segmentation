from scripts.main.generator import KerasGenerator
from tqdm import tqdm


def stupid_gen():
    gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_train2017.json',
                         dataset_dir='coco_dataset',
                         subset='train',
                         year='2017',
                         batch_size=4)
    gen.prepare()

    gen_try = gen.generate_batch()
    return gen_try


if __name__ == '__main__':
    gen = stupid_gen()
    for i in tqdm(range(10)):
        next(gen)
