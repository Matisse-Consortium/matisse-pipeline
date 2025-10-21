#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the Matisse pipeline GUI series
Copyright (C) 2017- Observatoire de la Côte d'Azur

Created on Tue Nov 19 13:50:34 2019
@author: ame

MATISSE BCD treatment tools

This software is a computer program whose purpose is to show oifits
files from the MATISSE instrument.

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.

You can use, modify and/ or redistribute the software under the
terms of the CeCILL license as circulated by CEA, CNRS and INRIA at
the following URL "http://www.cecill.info". You have a copy of the
licence in the LICENCE.md file.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""



import sys
import argparse
from libPostTools import mat_mergeByTplStart



if __name__ == '__main__':
    print("Python {}".format(sys.version))

    parser = argparse.ArgumentParser(description='Merging all oifits files from the same template into a single OiFits file per detector.')
    parser.add_argument('dirIn', default="",help='Input directory')
    parser.add_argument('--dirOut', default="MERGED",help='Output directory')
    parser.add_argument('--separateChopping', default="TRUE",help='speratating chopped and non-chopped exposure for LM band')
    try:
        args = parser.parse_args()
    except:
        print("\n\033[93mRunning mat_MergeAllOiFits.py --help to be kind with you:\033[0m\n")
        parser.print_help()
        print("\n     Example : python mat_MergeAllOiFits.py . ")
        sys.exit(0)

    if args.separateChopping=="TRUE":
        separateChopping=True
    else:
        separateChopping=False

    mergedData=mat_mergeByTplStart(args.dirIn,save=True,dirOut=args.dirOut,separateChopping=separateChopping)
