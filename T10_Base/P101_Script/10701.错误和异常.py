

# print(1 / 0)    # 0 不能作为除数，触发异常
'''
  File "D:\Ct_ iSpace\Wei\Python\MeiWei-Python\F20_Start\错误和异常.py", line 3, in <module>
    print(1 / 0)    # 0 不能作为除数，触发异常
ZeroDivisionError: division by zero
'''

# print(1 + '1')  # int 不能与 str 相加，触发异常
'''
  File "D:\Ct_ iSpace\Wei\Python\MeiWei-Python\F20_Start\错误和异常.py", line 10, in <module>
    print(1 + '1')  # int 不能与 str 相加，触发异常
TypeError: unsupported operand type(s) for +: 'int' and 'str'
'''

#print(1 + a)    # 变量未定义，触发异常
# while True print('Hello world')  # 语法错误，缺少了一个冒号


# 用户自定义异常
class MyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


try:
    raise MyError()
except MyError as e:
    print('My exception occurred, value:', e.value)
