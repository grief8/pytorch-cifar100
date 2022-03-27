#!/bin/bash
for constraint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
  echo constraint: $constraint
  python train.py -net mobilenet -gpu --arc-checkpoint ./checkpoints/oneshot/mobilenet/contraints-$constraint.json
#  python search.py --constraints $constraint --checkpoints ./checkpoints/oneshot/mobilenet/contraints-$constraint.json --model-path ./checkpoints/oneshot/mobilenet/contraints-$constraint.onnx
done;