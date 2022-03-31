#!/bin/bash
for constraint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
  model=$1
  lossType=$2
  dir=./checkpoints/oneshot/"${model}"/"${lossType}"
#  echo $constraint
  python eval_models.py --net "${model}" --arc-checkpoint "${dir}"/contraints-$constraint.json
done;
