# 2026-10-31 AI 주말 복습 문제

대상 범위: 2026-10-31 학습 코드 없음 → 10-23(트리 평가)·10-24(도구 호출)·10-25(토크나이저)·10-26(sparse 손실)·10-27(시계열 창)·10-28~10-30(`Conv1D` 계열)과 겹치지 않게, 아직 다루지 않은 **`keras2/keras70_1_주가맞추기.py`** 를 골랐습니다. 세 회사(아모레·하이닉스·삼성전자) 데이터를 날짜로 맞추고, 20일 창으로 잘라 세 갈래(LSTM 2 + Conv1D 1)를 `Concatenate`로 이어 삼성전자 시가를 맞히는 코드입니다. 함께 **날짜 정렬 · 정답이 미래를 미리 보는 문제 · 스케일 범위 · 파라미터 계산 · 콜백** 을 묻습니다.
문제 수: 20문제 (객관식 10문제, 주관식 10문제) — 10월의 마지막 주말이므로 20문제로 냅니다.
중복 주의: 2026-10-13 문서에서 같은 파일을 **정적 검토(실행 없이 코드 읽기)** 로 다뤘습니다. 오늘 문제는 실행으로 확인한 값(파라미터·스케일 범위·콜백 동작)과 **정답 누수** 쪽에 초점을 맞췄습니다.
비고: 문제에 나오는 숫자는 이 환경(tensorflow-cpu 2.15.1, keras 2.15.0, scikit-learn 1.4.2, pandas 2.1.4, NumPy 1.26.4, CPU 전용)에서 실제로 실행해 확인한 값입니다. 단, 원본 CSV(`_data/samsung/`)가 저장소에 없어 **원본 데이터 학습은 재현하지 못했고**, 같은 모양(1120행)의 합성 데이터로 구조·정렬·파라미터만 확인했습니다.

## 객관식

### 1. 아래 코드에서 `split_index` 값은 얼마일까요? (주석의 창 개수 880 + 221 = 1101개에서 역산한 값입니다.)

```python
split_size = 20
x3_windowed = split_xy(x3_filtered, split_size)   # 창 개수 1101
train_ratio = 0.8
split_index = int(len(x3_windowed) * train_ratio)
```

- A. 800
- B. 880
- C. 881
- D. 884

### 2. 창을 자른 뒤 정답을 붙이는 코드입니다. `y_windowed`의 i번째 값은 **어느 날의 시가**일까요?

```python
x3_windowed = split_xy(x3_filtered, 20)    # i번째 창 = i ~ i+19번째 줄
y_windowed  = y_data[20 - 1:]              # 시가 전체에서 앞 19개를 버림
```

- A. i번째 창의 **첫날** 시가
- B. i번째 창의 **마지막 날(19번째 줄)** 시가
- C. i번째 창 **다음 날** 시가
- D. i번째 창 첫날부터 19일 뒤 **종가**

### 3. 세 갈래 출력 `(None, 10)`, `(None, 10)`, `(None, 20)`을 이어 붙인 뒤 `Dense(70)`을 지납니다. 이 `Dense` 층의 파라미터 수는 몇 개일까요?

```python
merge1 = Concatenate()([output1, output21, output31])   # (None, 40)
merge2 = Dense(70, activation='relu')(merge1)
```

- A. 2,800개
- B. 2,870개
- C. 2,871개
- D. 700개

### 4. 아모레 갈래의 첫 LSTM 층입니다. 입력은 `(None, 20, 6)`입니다. 이 층의 파라미터 수는 몇 개일까요?

```python
input1 = Input(shape=(20, 6))
lstm1 = LSTM(20, return_sequences=True)(input1)
```

- A. 540개
- B. 2,080개
- C. 2,160개
- D. 2,240개

### 5. 삼성전자 갈래의 합성곱 층입니다. 커널 크기가 **1**인 점이 핵심입니다.

```python
conv3 = Conv1D(64, 1, activation='relu')(conv2)   # conv2는 (None, 20, 32)
```

- A. 2,048개이고, 날짜 20개를 한꺼번에 본다
- B. 2,112개이고, 날짜를 넘나들지 않아 "날짜마다 따로 적용되는 Dense"와 같다
- C. 2,112개이고, 이웃한 날짜 1개를 더 보므로 날짜 수가 19로 줄어든다
- D. 1,600개이고, 채널 축으로만 계산한다

### 6. `keras2/keras70_1_주가맞추기.py`는 스케일러를 **훈련 데이터로만** `fit`합니다. 상승 추세인 합성 데이터(훈련 80줄, 시험 20줄)로 확인했더니 시험 값 60개 중 **57개가 1을 넘었고 최댓값이 1.2104**였습니다. 이 결과를 가장 바르게 설명한 것은?

- A. 코드가 잘못되었으니 시험 데이터까지 넣어 `fit`해야 한다
- B. `MinMaxScaler`는 0~1 밖의 값을 잘라내므로 시험 값은 모두 1이 되어 문제가 없다
- C. `fit`을 훈련으로만 한 것은 올바르고, 시험 구간이 훈련에서 본 범위를 벗어날 수 있다는 사실을 감수해야 한다
- D. 시험 값이 1을 넘으면 `transform`이 자동으로 표준화로 바꾼다

### 7. 세 회사 파일의 줄을 맞추는 코드입니다. `isin`으로 거른 결과의 **줄 순서**는 어떻게 될까요?

```python
amore_csv = pd.read_csv("아모레.csv", encoding='euc-kr', thousands=',').sort_values(by='일자', ascending=True)
x3_dates  = x3['일자']
x1_filtered = x1[x1['일자'].isin(x3_dates)].reset_index(drop=True)
```

- A. 날짜 오름차순으로 자동 정렬된다
- B. 원본 파일의 줄 순서를 그대로 유지한다. 그래서 먼저 `sort_values`로 정렬해 두어야 한다
- C. 날짜 내림차순으로 자동 정렬된다
- D. `isin`은 순서를 보장하지 않으므로 실행할 때마다 달라진다

### 8. `validation_split=0.2`를 지정하고 `shuffle`은 건드리지 않았습니다. 검증에 쓰이는 데이터는 어느 부분일까요?

```python
model.fit([x1_train, x2_train, x3_train], y_train,
          epochs=100000, batch_size=42,
          validation_split=0.2, callbacks=[es, rlr, mcp])
```

- A. 무작위로 고른 20%
- B. 앞쪽 20%
- C. 뒤쪽 20%
- D. 정답 비율을 맞춘 층화 20%

### 9. 스케일링을 위해 3차원을 2차원으로 폅니다. 결과 모양은 무엇일까요?

```python
train_data = np.zeros((880, 20, 6))      # (샘플, 날짜, 특징)
train_2d = train_data.reshape(-1, train_data.shape[-1])
```

- A. `(880, 120)`
- B. `(17600, 6)`
- C. `(880, 20)`
- D. `(10560, 10)`

### 10. 학습이 끝난 뒤 창 1개로 예측합니다. `reshape(1, 20, 6)`이 필요한 이유로 가장 알맞은 것은?

```python
x1_pred = x1_filtered[-20:].to_numpy()                   # (20, 6)
x1_pred = scaler1.transform(x1_pred).reshape(1, 20, 6)
pred_scaled = model.predict([x1_pred, x2_pred, x3_pred]) # (1, 1)
pred_price  = y_scaler.inverse_transform(pred_scaled)
```

- A. 모델이 "샘플 여러 개"를 한꺼번에 받으므로, 창 1개도 '샘플 1개' 자리를 앞에 붙여야 한다
- B. `transform`이 3차원을 요구하기 때문이다
- C. `Dense` 층은 4차원 입력만 받기 때문이다
- D. `(20, 6)` 그대로 넣으면 날짜가 20개인 샘플 1개로 자동 해석되므로 아무 차이가 없다

## 주관식

### 11. 세 회사 주가 파일의 줄을 맞추는 과정을 설명해 보세요.

```python
amore_csv = pd.read_csv(csv_path + "아모레261012.csv", encoding='euc-kr', thousands=',').sort_values(by='일자', ascending=True)
...
x3_dates = x3['일자']
x1_filtered = x1[x1['일자'].isin(x3_dates)].reset_index(drop=True)
x1_filtered = x1_filtered.drop(['일자'], axis=1)
```

1. `sort_values(by='일자', ascending=True)`를 **왜 먼저** 해야 하는지, 이 줄이 없으면 데이터가 어떻게 되는지 적어 보세요.
2. `isin(x3_dates)`는 무엇을 하는 코드인가요? 세 회사의 줄 수가 다른 이유와 함께 설명해 보세요.
3. `reset_index(drop=True)`가 하는 일을, "중간 줄이 잘려 나간 표"를 예로 들어 설명해 보세요.
4. 같은 날짜가 파일에 두 번 들어 있으면 어떻게 될까요? 실제로 10줄짜리 표에 1줄을 복제해 `isin`으로 걸렀더니 결과가 **11줄**이 되었습니다. 이때 뒤에서 창을 자르면 무슨 문제가 생기는지 적어 보세요.

### 12. 이 코드에는 "정답이 미래를 미리 보는" 문제가 있습니다. 읽고 답해 보세요.

```python
x3_windowed = split_xy(x3_filtered, 20)     # 창 i = i ~ i+19번째 줄 (종가 포함)
y_windowed  = y_data[20 - 1:]               # 정답 = i+19번째 줄의 시가
```

1. i번째 창 안에는 그날의 **시가**와 **종가** 중 무엇이 들어 있나요? (삼성전자 갈래의 열 목록을 보고 답해 보세요.)
2. 하루의 시간 순서(시가 → … → 종가)를 생각할 때, 이 설정이 왜 "다음 날을 맞히는 문제"가 아닌지 설명해 보세요.
3. 합성 데이터로 확인한 값입니다. 창의 마지막 종가 하나로 선형 모델을 만들어 봤더니 **같은 날 시가**는 오차가 0.00, **다음 날 시가**는 197.04였습니다(단순히 전날 종가를 쓰면 222.26). 이 숫자가 1번·2번 답과 어떻게 이어지는지 설명해 보세요.
4. 정답을 "창 **다음** 날"로 바꾸는 코드를 적어 보세요. 이때 창 개수와 정답 개수가 1개 어긋나는데, 어떻게 맞추면 될까요?

### 13. 모델의 파라미터를 직접 계산해 보세요.

```python
input1 = Input(shape=(20, 6))
lstm1 = LSTM(20, return_sequences=True)(input1)
lstm2 = LSTM(128, return_sequences=True)(lstm1)
lstm3 = LSTM(64)(lstm2)
dense2 = Dense(80, activation='relu')(lstm3)
dense3 = Dense(30, activation='relu')(dense2)
output1 = Dense(10, activation='relu')(dense3)

merge1 = Concatenate()([output1, output21, output31])
merge2 = Dense(70, activation='relu')(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1, name='final_output')(merge3)
```

1. `LSTM` 파라미터 공식 `4 × ((입력 + 유닛) × 유닛 + 유닛)`을 써서 `LSTM(20)`, `LSTM(128)`, `LSTM(64)`를 각각 계산해 보세요.
2. `Dense(80)`, `Dense(30)`, `Dense(10)`을 계산해 보세요.
3. `Concatenate` 뒤의 `Dense(70)`, `Dense(5)`, `Dense(1)`을 계산해 보세요.
4. 이 모델 전체 파라미터가 **298,761개**입니다. 위에서 계산한 값들을 더해 이 숫자가 나오는지 확인해 보세요. (하이닉스 갈래와 삼성전자 Conv1D 갈래도 더해야 합니다.)

### 14. LSTM의 `4`는 무엇을 뜻할까요?

```python
lstm4 = LSTM(128, return_sequences=True)(lstm3)   # lstm3의 출력은 (None, 20, 63)
```

1. 공식 앞의 `4`가 가리키는 LSTM 내부의 네 가지 문(게이트)을 초등학생에게 설명하듯 쉽게 적어 보세요.
2. 이 층의 파라미터 수를 식과 함께 계산해 보세요.
3. `(None, 20, 63)`이 들어가는데 왜 `63`이 "날짜 20개"가 아니라 "유닛 수"로 쓰이는지 설명해 보세요.
4. `return_sequences=True`를 쓰는 층과 쓰지 않는 층의 출력 모양 차이를 적어 보세요. (마지막 LSTM만 `return_sequences`를 쓰지 않습니다.)

### 15. 삼성전자 갈래는 합성곱으로 만듭니다.

```python
input31 = Input(shape=(20, 5), name='input_samsung')
conv1 = Conv1D(8, 3, padding='same', activation='relu')(input31)
conv2 = Conv1D(32, 3, padding='same', activation='relu')(conv1)
conv3 = Conv1D(64, 1, activation='relu')(conv2)
conv4 = Conv1D(16, 1, activation='relu')(conv3)
flat = Flatten()(conv4)
dense34 = Dense(30, activation='relu')(flat)
output31 = Dense(20, activation='relu')(dense34)
```

1. 층을 지날 때마다 출력 모양이 어떻게 바뀌는지 표로 적어 보세요.
2. `padding='same'`을 쓴 층과 쓰지 않은 층에서 날짜 수가 각각 어떻게 되는지 적어 보세요. 만약 `padding`을 모두 빼면 `conv2`를 지난 뒤 날짜 수가 몇 개가 될까요?
3. `Flatten`을 지나면 값이 몇 개가 되나요? 그 뒤 `Dense(30)`의 파라미터도 계산해 보세요.
4. `Conv1D(64, 1)`의 파라미터가 `(1×32+1)×64 = 2,112`개입니다. 같은 32→64 `Dense` 층의 파라미터도 계산해서 두 값이 같은 이유를 설명해 보세요.

### 16. 스케일링 부분을 읽고 답해 보세요.

```python
def minMax_split(train_data, test_data):
    train_2d = train_data.reshape(-1, train_data.shape[-1])
    test_2d  = test_data.reshape(-1, test_data.shape[-1])
    scaler = MinMaxScaler()
    scaler.fit(train_2d)
    return scaler.transform(train_2d).reshape(train_data.shape), scaler.transform(test_2d).reshape(test_data.shape), scaler

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test  = y_scaler.transform(y_test.reshape(-1, 1))
```

1. 3차원을 2차원으로 펴는 이유를 `MinMaxScaler`가 받는 입력 모양과 함께 설명해 보세요.
2. `fit`, `transform`, `fit_transform`의 차이를 적어 보세요. `y_test`에는 왜 `fit_transform`이 아니라 `transform`을 써야 하나요?
3. 상승 추세 데이터에서 시험 값 60개 중 57개가 1을 넘고 최댓값이 1.2104였습니다. 왜 이런 일이 생기고, 모델 입장에서 무엇이 곤란한가요?
4. 예측한 0~1 값은 어떻게 원 단위 가격이 되나요? 쓰이는 함수 이름과, 그 함수가 `transform`의 정확한 반대인지 확인하는 방법을 적어 보세요.

### 17. 콜백 세 개를 설정했습니다.

```python
mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True,
                      filepath="./_save/keras70/kaggle_70_{epoch:04d}-{val_loss:.4f}.keras", verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=100, factor=0.1)
es  = EarlyStopping(monitor='val_loss', mode="min", patience=300, restore_best_weights=True)

model.fit(..., epochs=100000, batch_size=42, validation_split=0.2, callbacks=[es, rlr, mcp])
```

1. 세 콜백이 각각 하는 일을 한 문장씩 적어 보세요.
2. `patience=100`과 `patience=300`이 실제로 무엇을 기다리는지 설명해 보세요.
3. 파일 이름의 `{epoch:04d}`와 `{val_loss:.4f}`는 각각 무엇을 뜻하나요?
4. `epochs=100000`인데 실제 학습은 그만큼 오래 걸리지 않습니다. 왜일까요? 또 `batch_size=42`에 훈련 창이 880개면 1에폭에 가중치를 몇 번 고칠까요?

### 18. 학습 뒤 예측 코드를 검토해 보세요.

```python
x1_pred = x1_filtered[-20:].to_numpy()
x1_pred = scaler1.transform(x1_pred).reshape(1, 20, 6)
pred_scaled = model.predict([x1_pred, x2_pred, x3_pred])
pred_price = y_scaler.inverse_transform(pred_scaled)
print(f'내일 삼성전자 예측 시가: {pred_price[0][0]:,.0f}원')
```

1. 예측에 쓰는 창이 "마지막 20줄"인 것이 왜 자연스러운지 적어 보세요.
2. `reshape(1, 20, 6)`을 빼고 `(20, 6)`을 그대로 넣으면 무슨 일이 생길까요?
3. `inverse_transform`을 하지 않고 출력하면 어떤 숫자가 나오나요? 왜 그런가요?
4. 12번에서 본 정답 누수 때문에, 이 코드가 인쇄하는 값은 정말 "내일"의 시가일까요? 근거를 들어 판단해 보세요.

### 19. 코드 리뷰: 아래 설명 중 **실제 코드와 맞지 않는 것**을 찾아 고쳐 보세요. (최소 4개)

1. "창 20일, 줄 1120개면 창은 1100개다."
2. "`y_windowed = y_data[19:]`는 창 다음 날 시가를 정답으로 쓴다."
3. "삼성전자 갈래에서 시가를 뺐으니 정답 정보가 입력에 없다."
4. "주석의 `loss 0.3237`과 `228,903원`은 지금 다시 실행해도 같은 값이 나온다."
5. "`metrics`를 지정하지 않았으므로 `evaluate`는 loss와 정확도를 함께 돌려준다."
6. "`Conv1D(16, 1)`은 이웃한 날짜 1개를 더 보므로 날짜 수가 19가 된다."

### 20. pandas·NumPy 기초입니다.

```python
df = pd.read_csv("삼성전자261012.csv", encoding='euc-kr', thousands=',')
df = df.sort_values(by='일자', ascending=True).reset_index(drop=True)
arr = np.zeros((880, 20, 6)).reshape(-1, 6)
```

1. `encoding='euc-kr'`과 `thousands=','`는 각각 무슨 문제를 해결하나요? 두 번째 인자가 없으면 `1,234` 같은 값이 어떻게 읽힐까요?
2. `reset_index(drop=True)`와 `reset_index()`의 차이를 적어 보세요.
3. `arr.shape`는 무엇인가요? 계산 과정을 적어 보세요.
4. `np.arange(100).reshape(-1, 6)`을 실행하면 어떻게 될까요? 그 이유도 함께 적어 보세요.

## 안내

- 객관식은 보기의 알파벳만 적어도 됩니다.
- 주관식은 짧은 문장과 작은 예시를 함께 적어 보세요. 계산 문제는 계산 과정을 남겨 주세요.
- 1번은 `int(1101 × 0.8)`, 2번은 정답 위치(창의 마지막 날), 3번은 `Concatenate` 뒤 `Dense` 파라미터 `(40+1)×70`, 4번은 LSTM 공식 `4×((6+20)×20+20)`, 5번은 커널 1 합성곱의 의미, 6번은 훈련 전용 `fit`의 대가, 7번은 `isin`의 순서 유지, 8번은 `validation_split`의 위치, 9번은 `880×20`, 10번은 '샘플 1개' 차원을 묻습니다.
- 11번은 날짜 정렬·중복, 12번은 미래 정보 누수, 13·14번은 파라미터와 LSTM 게이트, 15번은 `padding='same'`과 커널 1, 16번은 스케일러 `fit`/`transform`/`inverse_transform`, 17번은 콜백, 18번은 예측 절차, 19번은 코드·주석 오류, 20번은 pandas·NumPy 기초를 묻습니다.
- 원본 CSV가 저장소에 없어 실제 학습·평가는 재현하지 못했습니다. 6번·12번의 실험 숫자는 같은 모양의 합성 데이터로 확인한 값이며, 실제 주가 데이터에서의 정확도 변화는 측정하지 않았습니다.
