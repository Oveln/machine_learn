# 返回大的那个
def max(x,y):
    if x>y :
        return x
    else:
        return y
    
# 返回list中所有偶数的平方组成的list
def work(list):
    ret = []
    for i in list:
        if i%2 == 0:
            ret.append(i*i)
    return ret

# 输出str到output.txt
def print_string_to_file(str):
    with open("output.txt", "w") as f:
        f.write(str)

# 返回所有奇数平方和
def work2(array):
    sum = 0
    for i in array:
        if i%2 == 1:
            sum+= i*i
    print(sum)