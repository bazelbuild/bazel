ar = [i for i in range(100000)]
target = 33333

def bench_binary_search():
    st, en = 0, len(ar) - 1
    for i in range(1000000000):
        if st > en:
            break
        mid = (st + en) // 2
        if ar[mid] == target:
            return mid
        elif ar[mid] > target:
            en = mid - 1
        else:
            st = mid + 1
    return -1

def bench_linear_search():
    for i in range(len(ar)):
        if ar[i] == target:
            return i
    return -1
