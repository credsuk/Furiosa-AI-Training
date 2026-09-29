# 누적 학습 진도

최종 갱신일: 2026-09-29

## 현재까지 확인한 범위

저장소 `main`의 200개 이상 Python 파일과 `keras01`부터 `keras52`까지의 학습 흐름을 최초 검토했습니다.

- Python 기본 문법: 변수, 연산, 실행
- NumPy 배열과 `shape`
- 벡터·행렬·다차원 배열
- TensorFlow/Keras `Sequential` 모델
- `Dense` 레이어와 `input_dim`
- 회귀 모델의 `compile`, `fit`, `evaluate`, `predict`
- `loss='mse'`, Adam optimizer
- epoch와 batch size
- 다중 입력·다중 출력 모델
- `train_test_split`, 학습 데이터와 테스트 데이터
- `random_state`, 데이터 섞기
- 산점도 시각화
- California Housing, Diabetes, Boston Housing 데이터셋
- `r2_score`, MSE, RMSE
- Pandas CSV 로딩, `index_col`, 결측치 제거, X/Y 분리
- 따릉이 데이터 전처리 및 제출용 데이터 구조
- validation set과 `validation_split`
- 학습 시간 측정과 학습 곡선
- 과적합 시각화
- EarlyStopping
- 이진 분류: sigmoid, binary crossentropy, accuracy
- 다중 분류: One-hot, softmax, categorical crossentropy
- 데이터 스케일링: MinMax, Standard, MaxAbs, Robust scaler
- 모델·가중치 저장 및 불러오기
- ModelCheckpoint
- Dropout과 정규화 방식
- CNN: Conv2D, Pooling, Flatten, GlobalAveragePooling
- 이미지 분류: MNIST, Fashion-MNIST, CIFAR10/100
- 이미지 파일을 NumPy 배열로 변환·저장·복원
- 이미지 증강: ImageDataGenerator
- optimizer와 learning rate 조정
- GPU 사용 여부와 학습 성능 비교

## 아직 확인이 필요한 부분

- 각 평가 지표의 의미와 좋은 방향
- 회귀와 분류의 문제 유형·출력층·손실 함수 선택
- 훈련 데이터와 테스트 데이터의 역할 차이
- validation set과 test set의 역할 차이
- `shape`와 `input_dim`의 정확한 관계
- 결측치 삭제와 대체의 선택 기준
- 과적합·과소적합과 검증 방법
- 모델 구조와 epoch·batch size를 정하는 근거
- 데이터 누수와 일반화 성능
- `sigmoid`와 `softmax`, binary/categorical crossentropy의 관계
- 스케일러는 train에 fit하고 test에는 transform만 해야 하는 이유
- Dropout, EarlyStopping, ModelCheckpoint의 역할 차이
- CNN의 shape 변화와 pooling
- 원본 이미지와 증강 이미지의 역할

## 답변 분석 기록

아직 제출된 복습 답변이 없습니다.

## 다음 복습 우선순위

1. 회귀·분류 문제의 출력층과 손실 함수
2. validation·test 분리와 과적합 대응
3. 스케일링과 데이터 누수 방지
4. CNN의 입력 shape·Conv2D·Pooling
5. 저장·복원·증강·optimizer 설정
