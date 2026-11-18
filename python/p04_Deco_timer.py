
# 실무에서는 다음고 같은 기능을 여러 함수에 공통으로 적용할 때 유용하다
# - 함수 실행시가 ㄴ측정
# - 로그 기록
# - 사용자 권한 검사
# - 예외 처리
# - 함수 호출 횟수 기록

import time

def timer(func):
    def wrapper():
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        print("걸린시간 : ", round(end - start, 2), "초")
    return wrapper

@timer
def hello():
    print("안녕하세요")
    time.sleep(1)

hello()

@timer
def study():
    print("NPU 공부중이다")
    time.sleep(1.5)

study()
