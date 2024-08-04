#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Pre-Traind Model to Standard CNN Version')
    parser.add_argument('--input', default='', type=str, metavar='PATH', required=True,
                        help='path to pre-trained checkpoint')
    parser.add_argument('--output', default='', type=str, metavar='PATH', required=True,
                        help='path to output checkpoint in general format')
    args = parser.parse_args()
    print(args)

    # load input
    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint['model']

    for m in list(state_dict._metadata.keys()):
        if m.startswith('') and m.startswith('module'):
            pass
        if m.startswith('module.student_encoder') and not m.startswith('module.student_encoder.mlps'):
            # remove prefix
            state_dict._metadata[m[len("module.student_encoder."):]] = state_dict._metadata[m]
        # delete renamed or unused k
        del state_dict._metadata[m]

    for m in list(state_dict._metadata.keys()):
        if m.startswith('fc') and m != 'fc1':
            del state_dict._metadata[m]
    for m in list(state_dict._metadata.keys()):
        if m.startswith('fc1'):
            state_dict._metadata[m.replace('1', '')] = state_dict._metadata[m]
            del state_dict._metadata[m]

    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.student_encoder') and not k.startswith('module.student_encoder.mlps'):
            # remove prefix
            state_dict[k[len("module.student_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    for k in list(state_dict.keys()):
        if k.startswith('fc') and not k.startswith('fc1.0'):
            del state_dict[k]
        if k.startswith('fc1.0'):
            state_dict[k.replace('1.0', '')] = state_dict[k]
            del state_dict[k]

    # make output directory if necessary
    output_dir = os.path.dirname(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # save to output
    torch.save({'model': state_dict}, args.output)
