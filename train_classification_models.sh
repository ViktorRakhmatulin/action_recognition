#!/bin/bash

# FPSs=60 30 15 10
# FPSs=40 24 20 12 8 6
# FPSs=1 2 4 0.5 6 8 12 20 24 40
# FPSs=120 60 40 24 20 12 8 6 4 2 1 0.5

FPSs=120

for FPS in $FPSs; do
for DIR in '-u' '-b'; do

# This command trains classification model
python train_classify.py --run-dir runs_classification \
    data/HDM05-122/HDM05-122-only-annot-subseq-fold-1-of-2.pkl\
    data/HDM05-122/HDM05-122-only-annot-subseq-fold-2-of-2.pkl\
    $DIR -f $FPS

done
done

