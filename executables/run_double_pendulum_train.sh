#!/bin/bash
python main.py --sys_name double_pendulum --train True --net_init kaiming_uniform \
    --hidden_vec 200 200 200 200 200 \
    --miles 600 1200 --gamma 0.1 --max_epoch 1800
