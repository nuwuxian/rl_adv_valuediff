#!/usr/bin/env bash
for i in 0; do
    python adv_train.py --env 2 > console_$i.txt &
    sleep 10
done
