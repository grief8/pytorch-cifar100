#!/bin/bash
for constraint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
  echo constraint: $constraint
#  python train.py -net mobilenet -gpu --arc-checkpoint ./checkpoints/oneshot/mobilenet/contraints-$constraint.json
#  search mobilenet
#  python search.py \
#  --constraints $constraint \
#  --checkpoints ./checkpoints/oneshot/mobilenet/logfunction/contraints-$constraint.json \
#  --model-path ./checkpoints/oneshot/mobilenet/logfunction/contraints-$constraint.onnx
#  search resnet18
  dir=./checkpoints/oneshot/resnet18/logfunction
  mkdir -p "${dir}"
  python search.py \
  --net resnet18 \
  --gpu \
  --worker-id 1 \
  --constraint $constraint \
  --checkpoints "${dir}"/contraints-$constraint.json \
  --model-path "${dir}"/contraints-$constraint.onnx

done;