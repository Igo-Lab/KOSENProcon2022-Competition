#!/bin/bash

source .venv/bin/activate
python -m kosenprocon
deactivate
echo "通信できないときは，"
echo "システム日時の確認をして，ずれてたら設定する(sudo datetimectl set-time "2022-10-13 13:37")"
echo "config.iniのトークンとconstant.py内のproxyの値を確認すること．"
