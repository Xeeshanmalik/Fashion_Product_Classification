import sys
import argparse
import ujson
from toolz.functoolz import compose
from functools import partial
import numpy as np

from common.pretty_json import pretty_json

image_labels = {}


def get_coding_length(item):

    image_labels[item['imageId']] = item['labelId']

    return item['labelId']


def one_hot_codes(unique, key, value):

    one_hot_code = np.zeros(len(unique), dtype=np.double)

    value = set(list(map(int, value)))

    indices = [i for i, e in enumerate(unique) if e in value]

    one_hot_code[indices] = 1

    return {'imageId': key, 'label': list(one_hot_code)}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fashion Product Classification')
    parser.add_argument('--input_metadata_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('--output_metadata_file', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help='The metadata file to process. Writes to the stdout by default')
    parser.add_argument('--class_info', type=str, required=True,
                        help='names of distinct classes')


    args = parser.parse_args()

    items = ujson.load(args.input_metadata_file)

    unique_coding = compose(get_coding_length)

    labels = []

    for index, item in enumerate(map(unique_coding, items['annotations'])):
        labels.append(item)

    unique_labels = sorted(list(map(int, set([item for sublist in labels for item in sublist]))))

    file= open(args.class_info, 'w')
    for item in unique_labels:
        file.write("{}\n".format(item))

    one_hot_coding = compose(partial(one_hot_codes, unique_labels))
    print('[')
    prev = None
    for item in map(one_hot_coding, image_labels.keys(), image_labels.values()):
        if prev is not None:
            print(',')
        print(pretty_json(item), end='')
        prev = item
    print('\n]')
