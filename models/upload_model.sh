# MODEL_ID로 구분되는 모델의 (1)config, (2)preprocessor, (3)학습된 parameter를 19번 서버에 업로드
# 사용법: ./upload_model.sh MODEL_ID

#!/bin/bash
model=$1
rsync -avr $model cdev19.nlpr::R/home1/irteam/apache/taewook/dd_model
