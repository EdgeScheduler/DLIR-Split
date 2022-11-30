#!/bin/bash

SHELL_PATH=$(cd $(dirname $0) && pwd )

RootPath=$SHELL_PATH/../

for model in "resnet50" "vgg19"
do
    for count in 2 3 4 
    do
        echo "split $model to $count"
        $RootPath/bin/release/DLIR-SPLIT -c $count -m $model -s 5 -p 30
    done
done