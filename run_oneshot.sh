#!/bin/bash
for constraint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
  echo constraint: $constraint
  model=resnet50
  lossType=logfunction
  dir=./checkpoints/oneshot/"${model}"/"${lossType}"
  #  search
  mkdir -p "${dir}"
  python search.py \
  --net "${model}" \
  --gpu \
  --pretrained \
  --worker-id 1 \
  --constraint $constraint \
  --arc-checkpoint "${dir}"/contraints-$constraint.json \
  --model-path "${dir}"/contraints-$constraint.onnx
#  retrain
#  python train.py --net "${model}" --gpu --worker-id 4 --loss-type "${lossType}" --arc-checkpoint "${dir}"/contraints-$constraint.json

done;