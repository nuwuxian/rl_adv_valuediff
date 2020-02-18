#!/usr/bin/env bash

for i in 0 1 2 3 4 5 6 7 8; do
    python adv_train.py --env 2 --seed $i > console_$i.txt &
    sleep 10
done
