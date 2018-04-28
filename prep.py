import argparse
import os
import sys
import multiprocessing
import ujson
from toolz.functoolz import compose, pipe
from functools import partial
from multiprocessing import Pool
import time
import shutil
import requests


def read_image(images, annotations):
    image_data = requests.get(images['url']).content
    with open(os.path.join(args.base_image_path, str(annotations['imageId']) + ".jpg"), "wb") as handler:
        handler.write(image_data)

    return {'imageId': images['imageId'], 'url': images['url'], 'labelId': annotations['labelId']}


def create_sets(base_image_path, path, item):

    image_id = item['imageId']
    labels = item['labelId']
    maxlen = len(labels)
    for index, label in enumerate(labels):
        if os.path.isfile(os.path.join(base_image_path, image_id) + '.jpg') and os.stat(os.path.join(base_image_path, image_id) + '.jpg').st_size > 128:
            new_image_name = label + '_' + image_id
            if index == maxlen - 1:
                shutil.move((os.path.join(base_image_path, image_id) + '.jpg'), os.path.join(path, label)+'/'
                            + new_image_name + '.jpg')
            else:
                shutil.copy((os.path.join(base_image_path, image_id) + '.jpg'), os.path.join(path, label)+'/' +
                        new_image_name + '.jpg')
        else:
            print('images_' + image_id + '.jpg', "does not exist")
    return {'image_id': image_id}


def create_dir(path, class_names):

    for class_name in class_names:
        class_name = str(class_name)
        if not os.path.isdir(os.path.join(path, class_name)):
            os.makedirs(os.path.join(path, class_name))
        else:
            shutil.rmtree(os.path.join(path, class_name))
            os.makedirs(os.path.join(path, class_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performs saliency detection')
    parser.add_argument('-tt', '--meta_file', required=True, type=argparse.FileType('r'))
    parser.add_argument('--base_image_path', type=str, required=True,
                        help='Path of the folder containing the thumbnail images.')
    parser.add_argument('-p', '--number_of_processes', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('-c', '--chunk_size', type=int, default=1)
    parser.add_argument('-l', '--limit', type=int, default=sys.maxsize)
    parser.add_argument('-tr', '--preparation_set_image_path', required=True, help='path to preparation '
                                                                                   'directory data generator '
                                                                                   'should follow '
                                                                                   'input/'
                                                                                   'set-type(train/test)'
                                                                                   '/class_name/'
                                                                                   'class_name_....jpg')

    parser.add_argument('--class_info', type=str, required=True, help='names of distinct classes')

    args = parser.parse_args()

    items = ujson.load(args.meta_file)
    file = open(args.class_info, 'r')
    distinct_classes = list(map(int,file.readlines()))

    create_dir(args.preparation_set_image_path, distinct_classes)

    preparation_set = compose(partial(create_sets,
                                      args.base_image_path,
                                      args.preparation_set_image_path),
                              read_image)

    p = Pool(args.number_of_processes)

    start = time.clock()

    for image in map(preparation_set, items['images'],  items['annotations']):
        print("Successfully Saved:", image['image_id'] + '.jpg')

    print("Total Time:", time.clock() - start)

