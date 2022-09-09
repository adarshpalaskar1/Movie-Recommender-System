from bisect import bisect_left

def binary_search_(array, to_find, low=0, high=0):
    if high == 0:
        high = len(array)

    pos = bisect_left(array, to_find, low, high) 

    if pos != high and array[pos] == to_find :
        return pos
    else:
        return -1