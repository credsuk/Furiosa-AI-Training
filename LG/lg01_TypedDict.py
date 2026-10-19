from typing import Dict, TypedDict


# Dict[str, str] > 이런 형태로 쓰겠다 / 타입 힌트 / 강제성이 없다
sample_dict : Dict[str, str] = {
    'name' : '김말똥',
    'age' : '39',
    'job' : '회사원',
}

# print(sample_dict)
# {'name': '김말똥', 'age': '39', 'job': '회사원'}

##############################################################

# 이렇게 구현하면 우선 껍데기만 정의한다
class Person(TypedDict):
    name : str
    age : int
    job : str


person = Person(name='홍길동', age=40, job='도적')
# print(person)
# {'name': '홍길동', 'age': 40, 'job': '도적'}

##############################################################

# 클래스들도 하나의 타입니다
# 이렇게도사용 가능하다
person2 : Person = {
    'name' : '김개똥',
    'age' : 20,
    'job' : '전사'
}

# print(person2)
# {'name': '김개똥', 'age': 20, 'job': '전사'}


