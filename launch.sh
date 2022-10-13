#!/bin/bash

source .venv/bin/activate
python -m kosenprocon
deactivate
echo "通信できないときはconfig.iniのトークンとconstant.py内のproxyの値を確認すること．"