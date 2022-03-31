#!/bin/bash
function run() {
  constraint=$1
  wid=$(echo "scale=0;  ($1*10)%8/1" | bc)
#  if test "$wid" -eq 3
#  then
#      wid=7
#  fi
  model=$2
  lossType=$3
  echo start "${model}" "${lossType}" "$constraint" $wid
  dir=./checkpoints/oneshot/"${model}"/"${lossType}"
  #  search
  mkdir -p "${dir}"
  python search.py \
  --net "${model}" \
  --gpu \
  --pretrained \
  --worker-id $wid \
  --epochs 100 \
  --batch-size 1024 \
  --loss-type "${lossType}" \
  --constraint "$constraint" \
  --arc-checkpoint "${dir}"/contraints-"$constraint".json \
  --model-path "${dir}"/contraints-"$constraint".onnx
  #  retrain
  python train.py \
  --net "${model}" \
  --gpu \
  --pretrained \
  --worker-id $wid \
  --batch-size 1024 \
  --loss-type "${lossType}" \
  --arc-checkpoint "${dir}"/contraints-"$constraint".json
}

for constraint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
  run $constraint "$1" "$2" &
done;