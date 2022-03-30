#!/bin/bash
function run() {
  constraint=$1
  wid=$(echo "scale=0;  ($1*10)%8/1" | bc)
  model=resnet50
  lossType=l2regularization
  echo start "${model}" "${lossType}" $constraint
  dir=./checkpoints/oneshot/"${model}"/"${lossType}"
  #  search
  mkdir -p "${dir}"
  python search.py \
  --net "${model}" \
  --gpu \
  --pretrained \
  --worker-id $wid \
  --epochs 200 \
  --batch-size 512 \
  --loss-type "${lossType}" \
  --constraint $constraint \
  --arc-checkpoint "${dir}"/contraints-$constraint.json \
  --model-path "${dir}"/contraints-$constraint.onnx
#  retrain
  python train.py \
  --net "${model}" \
  --gpu \
  --pretrained \
  --worker-id $wid \
  --batch-size 512 \
  --loss-type "${lossType}" \
  --arc-checkpoint "${dir}"/contraints-$constraint.json
}

for constraint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
  run $constraint &
done;