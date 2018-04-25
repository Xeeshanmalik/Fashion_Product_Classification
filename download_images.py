import argparse
import sys
import ujson
from toolz.functoolz import compose
import requests
import os
import multiprocessing


def read_image(item):

    image_data = requests.get(item['url']).content

    with open(os.path.join(args.output_dir, str(item['imageId']) + ".jpg"), "wb") as handler:
        handler.write(image_data)

    return item['imageId']


class FullPaths(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

def is_dir(dirname):

    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    else:
        return dirname

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fashion Product Classification')
    parser.add_argument('--input_metadata_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='The metadata file to process. Reads from stdin by default.')
    parser.add_argument('--output_dir', action=FullPaths, type=is_dir)

    args = parser.parse_args()

    items = ujson.load(args.input_metadata_file)

    download_images = compose(read_image)

    pool = multiprocessing.Pool()

    for item in pool.imap(download_images, items['images']):
        print(str(item) + ".jpg", " Downloaded Successfully")