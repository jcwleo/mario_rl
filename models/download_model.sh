# 19번 서버에서 MODEL_ID로 구분되는 모델의 (1)config, (2)preprocessor, (3)학습된 parameter를 다운로드
# 사용법: ./download_model.sh MODEL_ID

#!/bin/bash
model=$1
wget 'http://cdev19.nlpr.nhnsystem.com:11230/taewook/dd_model/'$model''
