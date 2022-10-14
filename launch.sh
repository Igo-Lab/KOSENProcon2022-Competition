#!/bin/bash

sudo sh -c 'echo N > /sys/kernel/debug/gpu.0/timeouts_enabled'
source .venv/bin/activate
python -m kosenprocon
deactivate
echo '通信できないときは，'
echo 'システム日時の確認をして，ずれてたら設定する(sudo timedatectl set-time "2022-10-13 13:37")'
echo 'config.iniのトークンとconstant.py内のproxyの値を確認すること．'
