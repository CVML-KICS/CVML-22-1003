
from pickletools import read_uint1
from queue import Queue

from winreg import QueryInfoKey
from collections import deque

que = Queue(100)


def get_10(n):
    global que
    if len(que) > n:
        return que[-10:]
    else:
        return que

def return_queue(arr, n):
    global que
    que = get_10(que)
    for i in arr:
        que.put(i)
        
    return que



