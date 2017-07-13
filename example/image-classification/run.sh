#!/bin/bash -e

if [ -z "$1" ]; then
    echo "usage: $0 <file contains all IPs>"
    exit -1
fi

if [ -z "$2" ]; then
    NUM_GPUS=1
else
    NUM_GPUS=$2
fi

export FRAMEWARK_DIR=~/framewark

cd $(dirname $0)

cp $1 hosts

WORKERS=$(wc -l hosts | cut -d " " -f1)

for model in 'alexnet:256:227' \
    'vgg16:16:224' 'resnet50:32:224' 'resnet101:16:224' \
    'resnet152:16:224' 'googlenet:128:224'
    do
        python benchmark.py --worker_file hosts --worker_count $WORKERS \
            --gpu_count $NUM_GPUS --networks $model
    done
