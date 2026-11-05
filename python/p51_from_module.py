from machine.car import drive
from machine.tv import watch

drive()
watch()

print("=========================")
from machine import car, tv
car.drive()
tv.watch()


print("============ test 폴더 =============")
from machine.test import car, tv
car.drive()
tv.watch()
