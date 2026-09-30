import os
key = os.getenv("OPENAI_API_KEY") # 이렇게 환경 변수를 불러올 수 있다

if key is None:
    print("OPENAI_API_KEY")
else :
    print("키 길이 : ", len(key))
    print("키 확인 : ", key[:8] + "...." + key[-4:])


    
