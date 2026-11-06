# keras11_2_diabetes 복사

from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import time



#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=530)

parameter = {

}

#2. 모델 구성

# 함수형 모델을 직접 만든다
def build_model(
        dropout=0.5, optimizer='adam', activation='relu', 
        node1=128, node2=64, node3=32, node4=15, node5=8, 
        lr=0.001, batch_size=64) :
    inputs = Input(shape=(10,), name='imputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(dropout)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(dropout)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(dropout)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs, )
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse',)
    return model


def create_hyperparameteer():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.2, 0.3, 0.4, 0.5]
    activation = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16]
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'dropout' : dropout,
        'activation' : activation,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,
    }

from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# 사이킥런과 텐서플로는 호화니 안되기 때문에 이렇게 wrappers를 시켜줘야 한다 
keras_model  = KerasRegressor(build_fn=build_model, verbose=1,)
model = RandomizedSearchCV(
        keras_model, create_hyperparameteer(), 
        cv=3, random_state=232, n_iter=10, verbose=1
        # , n_jobs=1 # GPU에서 함께 넣으면 터질 수 있음
    )

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=8,
    verbose=1,
    factor=0.8
)

start = time.perf_counter()
model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks = [es, rlr], validation_split=0.1)
end = time.perf_counter()


print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :", model.best_params_)

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

#4. 평가, 예측
print("model.scroe : ", model.score(x_test, y_test))
print("best_score : ", model.best_score_)

print("걸린시간 : ", round(end - start, 2), "초")


# loss :  2877.24609375