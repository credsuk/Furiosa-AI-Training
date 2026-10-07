from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


def evaluate(model, x_test, y_test, start_time, end_time, *, name=None):
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)

    r2 = r2_score(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)

    print(f"================= {name} ================")
    print("훈련 시간 :", round(end_time - start_time, 2), "초")
    print("loss :", loss[0])
    print("acc :", round(loss[1], 4))
    print("r2 :", r2)
    print("mse :", mse)