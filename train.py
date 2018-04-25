import argparse
import os
import ijson
import ujson
import time
from toolz.itertoolz import take
import sys
from itertools import islice



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Model For Classification')
    parser.add_argument('--input_metadata_file', type=str, required=True,
                         help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('--dev_meta_file', type=argparse.FileType('r'), required=True,
                        help='input metafile for validation')
    parser.add_argument('--class_file', type=str, required=True, help='class information')
    parser.add_argument('--gpu_state', type=int, default=True)
    parser.add_argument('-ih', '--im_height', required=True, type=int, help='Image height in pixels')
    parser.add_argument('-iw', '--im_width', required=True, type=int, help='Image width in pixels')
    parser.add_argument('-ic', '--im_channel', required=True, type=int, help='Image channel')
    parser.add_argument('-l', '--limit', type=int, default=sys.maxsize,
                        help='Limit the items for compute predictions for')

    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if args.gpu_state is not True:

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    with open(os.path.join(root_dir,args.input_metadata_file), 'rb') as json_file:
        objects = ijson.items(json_file, 'item')

        for line in islice(objects, 100000000):
            print(line['imageId'])
            print(line['label'])

    file = open(args.class_file, 'r')
    classes = file.readlines()

















