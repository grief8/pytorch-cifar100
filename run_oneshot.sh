#!/bin/bash
for constraint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
  echo constraint: $constraint
#  python train.py -net mobilenet -gpu --arc-checkpoint ./checkpoints/oneshot/mobilenet/contraints-$constraint.json
#  search
  model=mobilenet
  dir=./checkpoints/oneshot/"${model}"/logfunction
  mkdir -p "${dir}"
  python search.py \
  --net "${model}" \
  --gpu \
  --worker-id 1 \
  --constraint $constraint \
  --checkpoints "${dir}"/contraints-$constraint.json \
  --model-path "${dir}"/contraints-$constraint.onnx

done;