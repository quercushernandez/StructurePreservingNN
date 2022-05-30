#!/bin/bash
python main.py --sys_name viscoelastic --train True --net_init kaiming_uniform \
    --hidden_vec 50 50 50 50 50 \
    --miles 500 1000 --gamma 0.1 --max_epoch 1500 \
    --dset_norm False
