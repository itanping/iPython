"""
List（列表） 是 Python 中使用最频繁的数据类型。

列表是写在方括号 [] 之间、用逗号分隔开的元素列表。[:]遵循左闭右开原则。
列表中元素的类型可以不相同，它支持数字，字符串甚至可以包含列表（所谓嵌套）。
和字符串一样，列表同样可以被索引和截取，列表被截取后返回一个包含所需元素的新列表。
与字符串不一样的是，列表中的元素是可以改变的
"""
print('---------------------------------------------------------------------------------------------------------------')
test_list = ['abc', 111, 0.12345, '列表', True]
test_sub_list = [-1, 111, 1.111, -1.111]

# 元素索引
print(test_list)                  # 输出完整列表
print(test_list[0])               # 输出列表第一个元素
print(test_list[1:3])             # 从第二个开始输出到第三个元素
print(test_list[2:])              # 输出从第三个元素开始的所有元素
print(test_list[2:-2])            # 输出从第三个元素开始至倒数第二个的所有元素
print(test_sub_list * 2)          # 输出两次列表
print(test_list + test_sub_list)  # 连接列表
print(len(test_list))             # 列表大小

'''
['abc', 111, 0.12345, '列表', True]
abc
[111, 0.12345]
[0.12345, '列表', True]
[0.12345]
[-1, 111, 1.111, -1.111, -1, 111, 1.111, -1.111]
['abc', 111, 0.12345, '列表', True, -1, 111, 1.111, -1.111]
5
'''


print('---------------------------------------------------------------------------------------------------------------')
test_list = [1, 111, 2.2, 3, 4]
test_sub_list = [-1, 111, 1.111, -1.111]

# 内置函数
test_list.append(1)
print(test_list)                  # 把一个元素添加到列表的结尾。
test_list.insert(0, 1.1)          # 在指定位置插入一个元素。
test_list.extend(test_sub_list)   # 通过添加指定列表的所有元素来扩充列表。
print(test_list)                  # 在指定位置插入一个元素。
print(test_list.pop(1))           # 从列表的指定位置移除元素，并将其返回。如果没有指定索引，a.pop()返回最后一个元素。元素随即从列表中被移除。
print(test_list.index(111))       # 返回列表中第一个值为 x 的元素的索引。如果没有匹配的元素就会返回一个错误。
print(test_list.count(111))       # 返回 x 在列表中出现的次数。
test_list.remove(True)            # 删除列表中值为 x 的第一个元素。如果没有这样的元素，就会返回一个错误。
print(test_list)
test_list.reverse()               # 倒排列表中的元素。
print(test_list)
"""
[1, 111, 2.2, 3, 4, 1]
[1.1, 1, 111, 2.2, 3, 4, 1, -1, 111, 1.111, -1.111]
1
1
2
[1.1, 111, 2.2, 3, 4, -1, 111, 1.111, -1.111]
[-1.111, 1.111, 111, -1, 4, 3, 2.2, 111, 1.1]
"""


print('---------------------------------------------------------------------------------------------------------------')
test_list[4] = False                             # 修改列表元素值
test_sub_list[1:2] = [111, 111.111, 'TEST-111']  # 修改列表元素值
print(test_list)
print(test_sub_list)
test_list[2] = []                                # 将对应的元素值设置为 []
print(test_list)
"""
[-1.111, 1.111, 111, -1, False, 3, 2.2, 111, 1.1]
[-1, 111, 111.111, 'TEST-111', 1.111, -1.111]
[-1.111, 1.111, [], -1, False, 3, 2.2, 111, 1.1]
"""



print('---------------------------------------------------------------------------------------------------------------')
# 将列表当做堆栈使用
stack = [1, 2, 3, 4, 5]
stack.append(6)
print(stack)
print(stack.pop())
print(stack)
print(stack.pop())
print(stack.pop())
print(stack)

'''
[1, 2, 3, 4, 5, 6]
6
[1, 2, 3, 4, 5]
5
4
[1, 2, 3]
'''

print('---------------------------------------------------------------------------------------------------------------')
sites = ["BaiDu", "Google", "WiKi", "TaoBao"]
for site in sites:
    if len(site) < 5:
        print("WiKi!")
        break
    print("循环数据 " + site)
else:
    print("没有循环数据!")
print("完成循环!")

"""
循环数据 BaiDu
循环数据 Google
WiKi!
完成循环!
"""



print('---------------------------------------------------------------------------------------------------------------')
sites = ["BaiDu", "Google", "WiKi", "TaoBao"]
for i in range(len(sites)):
    print(i, sites[i])

"""
0 BaiDu
1 Google
2 WiKi
3 TaoBao
"""


print('---------------------------------------------------------------------------------------------------------------')
testList = ['abc', 111, 0.12345, '列表', True]
it = iter(testList)                  # 创建迭代器对象
print(next(it))                      # 输出迭代器的下一个元素
print(next(it))                      # 输出迭代器的下一个元素

"""
abc
111
"""

testList = ('北京', '上海', '广州', '深圳')
it = iter(testList)                 # 创建迭代器对象
for provence in it:
    print(provence, end=" ")

"""
北京 上海 广州 深圳 
"""

print()
print('---------------------------------------------------------------------------------------------------------------')
testList = {'Google', 'TaoBao', 'WeiXin', 'BaiDu'}
it = iter(testList)                 # 创建迭代器对象
while True:
    try:
        print(next(it).upper(), end=" ")
    except StopIteration:           # StopIteration 异常用于标识迭代的完成，防止出现无限循环的情况
        exit("对象迭代完了")

"""
WEIXIN BAIDU TAOBAO GOOGLE 
"""


