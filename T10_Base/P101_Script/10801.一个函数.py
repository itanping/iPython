# 可变对象实例
def chang_list(test_list):
    test_list.append([1, 2, 3, 4])  # 修改传入的列表
    print("函数内取值: %s，内存地址：%s" % (test_list, id(test_list)))
    return


# 调用 changList 函数
test_list = [10, 20, 30]
chang_list(test_list)
print("函数外取值: %s，内存地址：%s" % (test_list, id(test_list)))

print('---------------------------------------------------------------------------------------------------------------')


# 不可变对象实例
def change(a):
    print("2.1.函数内取值: %s，内存地址：%s" % (a, id(a)))  # 指向的是同一个对象
    a = 111
    print("2.2.函数内取值: %s，内存地址：%s" % (a, id(a)))  # 一个新对象
    return a


a = 1
print("1.1.函数外取值: %s，内存地址：%s" % (a, id(a)))
change(a)
print("1.2.函数外取值: %s，内存地址：%s" % (a, id(a)))
a = change(a)
print("1.3.函数外取值: %s，内存地址：%s" % (a, id(a)))

print('---------------------------------------------------------------------------------------------------------------')


# 默认参数
def print_info(name, age=66):  # 调用函数时，如果age没有传递参数，则会使用默认参数
    print("名字: %s，年龄: %s" % (name, age))
    return


print_info(age=50, name="Python")  # 使用关键字参数允许函数调用时参数的顺序与声明时不一致，因为 Python 解释器能够用参数名匹配参数值。
print_info(name="Python")  # 调用函数时，如果没有传递参数，则会使用默认参数。以下实例中如果没有传入 age 参数，则使用默认值。


# 不定长参数（**：字典形式）
def print_info_dict(name, age=66, **vardict):  # 不定长参数（**：字典形式）
    print("名字: %s，年龄: %s" % (name, age))
    print("不定长参数: ", vardict)
    return


print_info_dict(age=77, name="Python", id=111, sex='男')


# 不定长参数（*：元组形式）
def print_info_tuple(userid, *vartuple):  # 不定长参数（*：元组形式）
    print("输出: ", userid)
    print("不定长参数: ", vartuple)


print_info_tuple(1, 2, 3)

print('---------------------------------------------------------------------------------------------------------------')
# lambda 匿名函数
sum = lambda arg1, arg2: arg1 + arg2

# 调用sum函数
print("相加后的值为 : ", sum(1, 2))
print("相加后的值为 : ", sum(11, 22))

print('---------------------------------------------------------------------------------------------------------------')


# return 语句
def sum(arg1, arg2):
    total = arg1 + arg2
    print("函数内 : ", total)
    return total


# 调用sum函数
total = sum(10, 20)
print("函数外 : ", total)

if __name__ == '__main__':
    print('程序自身在运行')
else:
    print('程序被其它模块运行')
