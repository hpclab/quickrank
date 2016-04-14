#!/usr/bin/env python
# coding=utf-8

import argparse
import sys
import logging
import xml.etree.ElementTree as ET


def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    parser = argparse.ArgumentParser(description='Merge starting model with weights found by a pruning strategy...')
    parser.add_argument("-w", "--weights", dest="weights_filename", required=True,
                        help="model with weights to import", type=argparse.FileType('r'))
    parser.add_argument("-m", "--model", dest="model_filename", required=True,
                        help="model with weights to overwrite", type=argparse.FileType('r'))
    parser.add_argument("-d", "--destination", dest="destination_model_filename", required=True,
                        help="destination model with fixed weights", type=argparse.FileType('wb', 0))
    args = parser.parse_args()

    weights = []
    forest = ET.parse(args.weights_filename)
    for tree in forest.iter('tree'):
        index = int(tree.find('index').text)
        weight = float(tree.find('weight').text)
        assert(len(weights) == index-1)
        weights.append(weight)

    skipped = 0
    model = ET.parse(args.model_filename)
    ensemble = model.find('ensemble')
    for tree in ensemble.findall('tree'):
        index = int(tree.get('id'))
        if weights[index-1] == 0:
            ensemble.remove(tree)
            skipped += 1
        else:
            tree.set('id', str(index - skipped))
            tree.set('weight', str(weights[index-1]))

    model.write(args.destination_model_filename)

if __name__ == '__main__':
    sys.exit(main())