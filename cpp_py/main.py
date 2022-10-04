import ctypes as ct
from re import S

if (__name__ == "__main__"):
    libc = ct.cdll.LoadLibrary("./main.so")

    unko = input("入力")
    print(libc.kuso(int(unko)))

    s=ct.c_byte()
    libc.kass(s)
    print(s)

    

#print(__name__)
#print(libc.add(1,2))