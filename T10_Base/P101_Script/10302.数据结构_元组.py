"""
元组（tuple）与列表类似，不同之处在于元组的元素不能修改。元组写在小括号 () 里，元素之间用逗号隔开。

与字符串一样，元组的元素不能修改。
"""
print('---------------------------------------------------------------------------------------------------------------')
tupleTest = ()     # 空元组
print(tupleTest)
tupleTest = (20,)  # 一个元素，需要在元素后添加逗号
print(tupleTest)

tupleTest = ('abc', 111, 0.12345, '列表', True)
tinyTuple = (123, 'TEST')

print(tupleTest)              # 输出完整元组
print(tupleTest[0])           # 输出元组的第一个元素
print(tupleTest[1:3])         # 输出从第二个元素开始到第三个元素
print(tupleTest[2:])          # 输出从第三个元素开始的所有元素
print(tinyTuple * 2)          # 输出两次元组
print(tupleTest + tinyTuple)  # 连接元组
tupleList = (tupleTest, tinyTuple)
print(tupleList)
# tupleTest[0] = 11             # 修改元组元素的操作是非法的：TypeError: 'tupleTest' object does not support item assignment

'''
()
(20,)
('abc', 111, 0.12345, '列表', True)
abc
(111, 0.12345)
(0.12345, '列表', True)
(123, 'TEST', 123, 'TEST')
('abc', 111, 0.12345, '列表', True, 123, 'TEST')
(('abc', 111, 0.12345, '列表', True), (123, 'TEST'))
'''
