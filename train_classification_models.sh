#!/bin/bash

# Runs LSTM hyperparams training/evaluation pipeline (biderectional /unidir LSTM units, Linear units)
#for DIR in '-u' '-b'
#do
#  for st in 1 2
#    do
#      for lr in 1 2
#      do
#          echo $DIR $st $lr
#          python train_classify.py --run-dir runs_classification \
#              data/HDM05-15/HDM05-15-part1.pkl\
#              data/HDM05-15/HDM05-15-part2.pkl $DIR -s $st -l $lr
#        done
#    done
#done

#Run HDM05-122 dataset
#python train_classify.py --run-dir runs_classification \
#    data/HDM05-122/HDM05-122-only-annot-subseq-fold-1-of-2.pkl\
#    data/HDM05-122/HDM05-122-only-annot-subseq-fold-2-of-2.pkl -e 1
