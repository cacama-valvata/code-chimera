#Source: https://stackoverflow.com/questions/46610160/os-ftruncate-is-showing-some-error-while-execution-error-like-typeerror-a-byt
import os,sys

sr=os.open("sri.txt",os.O_RDWR|os.O_CREAT)

os.write(sr,"This is the test-This is test")

os.ftruncate(sr,10)

os.lseek(sr,0,0)

str=os.read(sr,100)

print("Read string is:",str)

os.close(sr)

print("closed the file successfully!!")