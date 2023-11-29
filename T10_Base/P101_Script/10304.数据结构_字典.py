"""
字典（dictionary）是Python中另一个非常有用的内置数据类型。

列表是有序的对象集合，字典是无序的对象集合。两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。
字典是一种映射类型，字典用 { } 标识，它是一个无序的 键(key) : 值(value) 的集合。
键(key)必须使用不可变类型。
在同一个字典中，键(key)必须是唯一的。
"""

print('---------------------------------------------------------------------------------------------------------------')
dict = {}  # 空字典
dict['one'] = "1 - Python学习"
dict[2] = "2 - Python编程"
tinyDict = {'name': 'Python', 'code': 1, 'site': 'www.baidu.com'}

print(dict['one'])        # 输出键为 'one' 的值
print(dict[2])            # 输出键为 2 的值
print(tinyDict)           # 输出完整的字典
print(tinyDict.keys())    # 输出所有键
dictVal = tinyDict.values()
print(dictVal)  # 输出所有值

'''
1 - Python学习
2 - Python编程
{'name': 'Python', 'code': 1, 'site': 'www.baidu.com'}
dict_keys(['name', 'code', 'site'])
dict_values(['Python', 1, 'www.baidu.com'])
'''