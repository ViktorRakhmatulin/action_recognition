#!/bin/bash

# Runs LSTM hyperparams training/evaluation pipeline
for DIR in '-u' '-b'
do
  for st in 1 2
    do
      for lr in 1 2
      do
          echo $DIR $st $lr
          python train_classify.py --run-dir runs_classification \
              data/HDM05-15/HDM05-15-part1.pkl\
              data/HDM05-15/HDM05-15-part2.pkl $DIR -s $st -l $lr
        done
    done
done
