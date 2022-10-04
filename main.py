import ctypes as ct

if (__name__ == "__main__"):
    libc = ct.cdll.LoadLibrary("./libmyadd.so")

print(libc.add(1, 2))
    

#print(__name__)
#print(libc.add(1,2))