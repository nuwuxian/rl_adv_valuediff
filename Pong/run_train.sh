#!/usr/bin/env bash

for i in 0 1 2 3; do
    python play_pong_train.py --hyper_index $i > console_$i.txt &
    sleep 10
done
