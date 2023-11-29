"""
Python 推导式

Python 支持各种数据结构的推导式：
列表(list)推导式
字典(dict)推导式
集合(set)推导式
元组(tuple)推导式
"""


print('---------------------------------------------------------------------------------------------------------------')
"""
列表推导式
[表达式 for 变量 in 列表] 
或
[表达式 for 变量 in 列表 if 条件]
"""
names = ['Google', 'Taobao', 'WeiXin', 'Facebook', 'Zhihu', 'Baidu']

# 过滤掉长度小于或等于5的字符串列表，并将剩下的转换成大写字母
new_names = [name.upper() for name in names if len(name) > 5] 
print(new_names)

multiples = [i for i in range(30) if i % 3 == 0]
print(multiples)

'''
['GOOGLE', 'TAOBAO', 'WEIXIN', 'FACEBOOK']
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
'''


print('---------------------------------------------------------------------------------------------------------------')
"""
字典推导式
{key_expr: value_expr for value in collection}
或
{key_expr: value_expr for value in collection if condition}
"""
listDemo = ['Google', 'Taobao', 'WeiXin', 'Facebook', 'Zhihu', 'Baidu']

# 将列表中各字符串值为键，各字符串的长度为值，组成键值对
newDict = {key: len(key) for key in listDemo if len(key) > 5}
newDict2 = {key for key in listDemo}
print(newDict)
print(newDict2)

'''
{'Google': 6, 'Taobao': 6, 'WeiXin': 6, 'Facebook': 8}
{'Google', 'Zhihu', 'Baidu', 'Taobao', 'WeiXin', 'Facebook'}
'''


print('---------------------------------------------------------------------------------------------------------------')
"""
集合推导式
{expression for item in Sequence}
或
{expression for item in Sequence if conditional}
"""
listDemo = ['Google', 'Taobao', 'WeiXin', 'Facebook', 'Zhihu', 'Baidu']
newSet = {value.upper() for value in listDemo if len(value) > 5}
print(newSet)

# 判断不是 abc 的字母并输出
listDemo = {x for x in 'abracadabra' if x not in 'abc'}
print(listDemo)

'''
{'WEIXIN', 'FACEBOOK', 'GOOGLE', 'TAOBAO'}
{'r', 'd'}
'''


print('---------------------------------------------------------------------------------------------------------------')
"""
元组推导式（生成器表达式）
元组推导式可以利用 range 区间、元组、列表、字典和集合等数据类型，快速生成一个满足指定需求的元组。
元组推导式和列表推导式的用法也完全相同，只是元组推导式是用 () 圆括号将各部分括起来，而列表推导式用的是中括号 []，另外元组推导式返回的结果是一个生成器对象。

(expression for item in Sequence)
或
(expression for item in Sequence if conditional)
"""
listDemo = ['Google', 'Taobao', 'WeiXin', 'Facebook', 'Zhihu', 'Baidu']
newTuple = (value.upper() for value in listDemo if len(value) > 5)
print(newTuple)
print(tuple(newTuple))

listDemo = (x for x in range(1, 10))
print(listDemo)
print(tuple(listDemo))

'''
<generator object <genexpr> at 0x00F9A108>
('GOOGLE', 'TAOBAO', 'WEIXIN', 'FACEBOOK')
<generator object <genexpr> at 0x02EDEB18>
(1, 2, 3, 4, 5, 6, 7, 8, 9)
'''
