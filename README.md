## GREET 모델의 성능 높이기
### GREET폴더
원래 코드에 discriminator 정확도를 계산하는 함수만 추가되어있음.

## 영규
### idea11
예측한 weight를 반올림하는 방법

### random_gcl
gcl을 학습시키지 않음. (어떻게 되는지 확인용)

### pexp5
rexp2에서 concat한 embedding들의 차원을 줄이는 과정 하나 추가

## 나은
### rexp2
TEST에서 평균내지말고 4개 다 concat, rank loss 구할 때도