#!/bin/bash

# build the docker image

docker build -t efficient_classification .

# run the training

docker run -m=25G --shm-size=20G \
    -v ./$1:/efficient_classification/data \
    efficient_classification \
    --training_data_path data/$2/ \
    --testing_data_path data/$3/ \
    --testing_data_solution_path data/$4/ \
    --batch_size 256 \
    --epochs 6 \
    --lr 0.01 \
    --train \
    --test 


