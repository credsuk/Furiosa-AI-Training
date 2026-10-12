import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, LSTM, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler


#1. 데이터
csv_path = "./_data/samsung/"
amore_csv = pd.read_csv(csv_path + "아모레261012.csv", encoding='euc-kr', thousands=',').sort_values(by='일자', ascending=True)
hynix_csv = pd.read_csv(csv_path + "하이닉스261012.csv", encoding='euc-kr', thousands=',').sort_values(by='일자', ascending=True)
samsung_csv = pd.read_csv(csv_path + "삼성전자261012.csv", encoding='euc-kr', thousands=',').sort_values(by='일자', ascending=True)

# print(amore_csv.columns)
# Index(['일자', '시가', '고가', '저가', '종가', '대비', 'Unnamed: 6', '등락률', '거래량','금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],dtype='str')

x1 = amore_csv[['일자', '시가', '종가', '등락률', '개인', '기관', '외국계']]
x2 = hynix_csv[['일자','시가', '종가', '등락률', '개인', '기관', '외국계']]
x3 = samsung_csv[['일자','종가', '등락률', '개인', '기관', '외국계']]
# print(x1.shape, x2.shape, x3.shape) # (2060, 6) (1300, 6) (1120, 5)


# x3 일자 데이터 추출
x3_dates = x3['일자']

# x3와 동일한 일자 데이터가 아니면 삭제
# 삭제후 데이터가 꼬여 있을 수 있으니 reset_index() > 인덱스 번호 0부터 재정렬
x1_filtered = x1[x1['일자'].isin(x3_dates)].reset_index(drop=True)
x2_filtered = x2[x2['일자'].isin(x3_dates)].reset_index(drop=True)
x3_filtered = x3.reset_index(drop=True)

# 불필요 일자 데이터 삭제
x1_filtered = x1_filtered.drop(['일자'], axis=1)
x2_filtered = x2_filtered.drop(['일자'], axis=1)
x3_filtered = x3_filtered.drop(['일자'], axis=1)
# print(x1_filtered.shape, x2_filtered.shape, x3_filtered.shape) # (1120, 6) (1120, 6) (1120, 5)


y_data = samsung_csv['시가'].to_numpy()
# print(y_data.shape) # (1120,)

# 시계열 데이터로 변환
def split_xy(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

split_size = 20

# 시계열 데이터로 자른다
x1_windowed = split_xy(x1_filtered, split_size)
x2_windowed = split_xy(x2_filtered, split_size)
x3_windowed = split_xy(x3_filtered, split_size)
y_windowed = y_data[split_size - 1 :]


# train, test 데이터 분리
train_ratio = 0.8
split_index = int(len(x1_windowed) * train_ratio)

# X 데이터 분할
x1_train, x1_test = x1_windowed[:split_index], x1_windowed[split_index:]
x2_train, x2_test = x2_windowed[:split_index], x2_windowed[split_index:]
x3_train, x3_test = x3_windowed[:split_index], x3_windowed[split_index:]

# Y 데이터 분할
y_train, y_test = y_windowed[:split_index], y_windowed[split_index:]


def minMax_split(train_data, test_data):
    # 3차원 -> 2차원으로 변환하여 스케일링
    train_2d = train_data.reshape(-1, train_data.shape[-1])
    test_2d = test_data.reshape(-1, test_data.shape[-1])

    scaler = MinMaxScaler()
    scaler.fit(train_2d)  # 반드시 Train 데이터로만 fit!
    
    train_scaled = scaler.transform(train_2d)
    test_scaled = scaler.transform(test_2d)

    # 다시 3차원으로 복원
    return (
        train_scaled.reshape(train_data.shape),
        test_scaled.reshape(test_data.shape),
        scaler,
    )

# X 데이터 스케일링 적용
x1_train, x1_test, scaler1 = minMax_split(x1_train, x1_test)
x2_train, x2_test, scaler2 = minMax_split(x2_train, x2_test)
x3_train, x3_test, scaler3 = minMax_split(x3_train, x3_test)

# print(x1_train.shape, x1_test.shape) # (880, 20, 6) (221, 20, 6)
# print(x2_train.shape, x2_test.shape) # (880, 20, 6) (221, 20, 6)
# print(x3_train.shape, x3_test.shape) # (880, 20, 5) (221, 20, 5)


# y값이 너무 커서 해당 값도 스케일링
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test = y_scaler.transform(y_test.reshape(-1, 1))
# print(y_train.shape, y_test.shape) # (880, 1) (221, 1)

# exit()

#2. 모델 구성

# 2-1. 모델 구성
input1 = Input(shape=(20, 6), name='input_amore')
lstm1 = LSTM(20, return_sequences=True)(input1)
lstm2 = LSTM(128, return_sequences=True)(lstm1)
lstm3 = LSTM(64)(lstm2)
dense2 = Dense(80, activation='relu')(lstm3)
dense3 = Dense(30, activation='relu')(dense2)
output1 = Dense(10, activation='relu')(dense3)

# 2-2.  모델 구성
input21 = Input(shape=(20, 6), name='input_hynix')
lstm21 = LSTM(63, return_sequences=True)(input21) 
lstm22 = LSTM(128, return_sequences=True)(lstm21)
lstm23 = LSTM(36)(lstm22)
dense22 = Dense(80, activation='relu')(lstm23)
dense23 = Dense(30, activation='relu')(dense22)
output21 = Dense(10, activation='relu')(dense23)

# 2-3.  모델 구성
input31 = Input(shape=(20, 5), name='input_samsung')
conv1 = Conv1D(8, 3, padding='same', activation='relu')(input31)
conv2 = Conv1D(32, 3, padding='same', activation='relu')(conv1)
conv3 = Conv1D(64, 1, activation='relu')(conv2)
conv4 = Conv1D(16, 1, activation='relu')(conv3)
flat = Flatten()(conv4)
dense34 = Dense(30, activation='relu')(flat)
output31 = Dense(20, activation='relu')(dense34)

# 2-4. 모델 구성
merge1 = Concatenate()([output1, output21, output31])
merge2 = Dense(70, activation='relu')(merge1)
merge3 = Dense(5, activation='relu')(merge2)

last_output = Dense(1, name='final_output')(merge3)

model = Model(inputs=[input1, input21, input31], outputs=last_output)

# model.summary()
# exit()

#3. 컴파일, 훈련
learning_rate = 0.004
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate),)


mcp = ModelCheckpoint(
    monitor='val_loss', # 모니터되는 데이터
    mode='auto', # 자동으로
    save_best_only=True, # 최고의 값만 저장해라
    # save_weights_only=True,
    filepath="./_save/keras70/kaggle_70_{epoch:04d}-{val_loss:.4f}.keras", # 파일 경로
    verbose=1, # 보여줄지 안보여줄지 확인하는 항목
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=100,
    verbose=1,
    factor=0.1 # 나누는 기준 기본값은 0.1
)

es = EarlyStopping(
    monitor='val_loss',
    mode="min",
    patience=300,
    restore_best_weights=True,
)

start_time = time.time()
model.fit([x1_train, x2_train, x3_train], y_train, 
        epochs=100000,
        batch_size=42,
        validation_split=0.2,
        # callbacks=[es, rlr],
        callbacks=[es, rlr, mcp],
        )
end_time = time.time()


#4. 평가 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
# y_predict = model.predict([x1_test, x2_test, x3_test])

print("================= keras70_1_주가맞추기 ================")
print("훈련 시간 :", round(end_time - start_time, 2), "초")
print("loss :", loss)


x1_pred = x1_filtered[-20:].to_numpy()  # shape: (20, 6)
x2_pred = x2_filtered[-20:].to_numpy()  # shape: (20, 6)
x3_pred = x3_filtered[-20:].to_numpy()  # shape: (20, 5)

x1_pred = scaler1.transform(x1_pred).reshape(1, 20, 6)
x2_pred = scaler2.transform(x2_pred).reshape(1, 20, 6)
x3_pred = scaler3.transform(x3_pred).reshape(1, 20, 5)

pred_scaled = model.predict([x1_pred, x2_pred, x3_pred])

pred_price = y_scaler.inverse_transform(pred_scaled)
print(f'내일 삼성전자 예측 시가: {pred_price[0][0]:,.0f}원')


# ================= keras70_1_주가맞추기 ================
# 훈련 시간 : 139.4 초
# loss : 0.323711633682251
# 내일 삼성전자 예측 시가: 228,903원