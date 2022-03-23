import argparse

import pandas as pd
import re
import pickle

import numpy as np

from itertools import groupby, chain

from tqdm import tqdm
import cv2 as cv
import pickle
from scipy.spatial.transform import Rotation as R


def drop(data_3d):
    data_pixels = np.zeros((data_3d.shape[0], 31, 2), dtype=np.float32)
    for index, frame in enumerate(data_3d):
        data_pixels[index] = frame[:, :-1]

    return data_pixels


def to_pixel(data_3d):
    intr_matrix = np.load('intrinsic_calibration_matrix.npy')
    intr_matrix[0, 2] = np.round(intr_matrix[0, 2])
    intr_matrix[1, 2] = np.round(intr_matrix[1, 2])
    dist = np.load('distortion_coeffs.npy')
    camera_rot = np.eye(3)
    rot_vec = R.from_matrix(camera_rot).as_rotvec()
    camera_trans = np.array([0., 0., 35])

    data_pixels = np.zeros((data_3d.shape[0], 31, 2), dtype=np.float32)

    for index, frame in enumerate(data_3d):
        pixels = cv.projectPoints(frame, rot_vec, camera_trans, intr_matrix, dist)[0].astype(np.int).astype(
            np.float32).reshape((31, 2))
        data_pixels[index] = pixels

    return data_pixels


def get_sequences(fle):
    with open(fle) as f:
        grps = groupby(f, key=lambda x: x.lstrip().startswith("#objectKey"))
        for k, v in grps:
            if k:
                yield list(chain([next(v)], (next(grps)[1])))  # all lines up to next #objectKey


def parse_data_line(line):
    values = re.split('; |, | ', line)
    values = map(float, values)
    values = np.fromiter(values, dtype=np.float32)
    values = values.reshape(-1, 3)
    return values


def parse_sequence(lines):

    header = lines[0]
    header_regexp = r'.*\s((\d+)_(\d+)_(\d+)_(\d+))'
    matches = re.match(header_regexp, header)
    assert matches is not None, "Error parsing data header: %s" % header
    attributes = matches.groups()
    sample_id = attributes[0]
    seq_id, action_id, start_frame, duration = map(int, attributes[1:])
    lines = lines[2:]  # discard header

    data = [parse_data_line(line) for line in lines]
    data = np.stack(data)
    dropped_data = drop(data)
    # data_pixel = to_pixel(data)
    # print(data.shape)

    sequence = dict(
        id=sample_id,
        seq_id=seq_id,
        action_id=action_id,
        start_frame=start_frame,
        duration=duration,
        data=dropped_data
    )
    return sequence


def get_ids_to_keep(split_file, format='list', train=True):

    if format == 'list':
        with open(split_file, 'rt') as f:
            ids = set(map(str.rstrip, f.readlines()))

    elif format == 'csv':
        ids = set(pd.read_csv(split_file, header=None).iloc[0])

    elif format == 'petr':
        with open(split_file, 'rt') as f:
            lines = f.readlines()

        idx = 1 if train else 4
        ids = map(int, lines[idx].rstrip('\n ,').split(','))
        ids = set(ids)

    return ids


def parse_annotated_sequence(lines, annotations):
    seq_id = int(lines[0].split(' ')[-1])
    duration = int(lines[1].split(';')[0])
    annotations = [a for a in annotations if a['seq_id'] == seq_id]
    # remove 'others' class from HDM05-15
    annotations = [a for a in annotations if a['action_id'] != 14]
    for a in annotations:
        if 'data' in a:
            del a['data']

    lines = lines[2:]  # discard header

    data = [parse_data_line(line) for line in lines]
    data = np.stack(data)
    sequence = dict(
        seq_id=seq_id,
        annotations=annotations,
        duration=duration,
        data=data
    )
    return sequence

def load_annotations(annot_file, format, train=True):
    if format == 'pkl':
        with open(annot_file, 'rb') as infile:
            annotations = pickle.load(infile)

    elif format == 'petr':
        with open(annot_file, 'rt') as infile:
            lines = infile.readlines()

        idx = 7 if train else 10
        ids = lines[idx].rstrip('\n ,').split(',')

        def parse_annotation(a):
            fields = a.strip().split('_')
            fields = map(int, fields)
            names = ('seq_id', 'action_id', 'start_frame', 'duration')
            return dict(zip(names, fields))

        annotations = [parse_annotation(i) for i in ids]

    return annotations


def main(args):
    print(args)
    sequences = get_sequences(args.data)

    if args.annotations:  # parse parent sequences containing multiple annotations
        annotations = load_annotations(args.annotations, args.af, args.train)
        parsed = (parse_annotated_sequence(seq, annotations) for seq in sequences)
    else:  # parse single annotated sequences
        parsed = (parse_sequence(seq) for seq in sequences)

    if args.split:
        key = 'seq_id' if args.annotations else 'id'
        ids_to_keep = get_ids_to_keep(args.split, args.sf, args.train)
        parsed = filter(lambda x: x[key] in ids_to_keep, parsed)

    parsed = list(tqdm(parsed))
    with open(args.parsed_data, 'wb') as outfile:
        pickle.dump(parsed, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse motion data')
    parser.add_argument('data', help='path to data file (in textual format)')
    parser.add_argument('-s', '--split', help='path to optional split file (in textual format)')
    parser.add_argument('-a', '--annotations', help='path to annotations file for parent sequences')
    parser.add_argument('--sf', '--split-format', choices=['list', 'csv', 'petr'], default='list', help='split format')
    parser.add_argument('--af', '--annot-format', choices=['pkl', 'petr'], default='pkl', help='annotation format')
    parser.add_argument('--test', action='store_false', dest='train', help='whether to save train or test annotations (for \'petr\' format only)')
    parser.add_argument('parsed_data', help='output file with parsed data file (in Pickle format)')
    parser.set_defaults(train=True)
    args = parser.parse_args()
    main(args)
