"""
Note the reduce is now in the functools module.
See https://stackoverflow.com/questions/181543/what-is-the-problem-with-reduce
and 
https://www.artima.com/weblogs/viewpost.jsp?thread=98196
for a detailed explanation from Guido himself.

"""
from functools import *

def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:      
        a, b = b, a % b
    return a

def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)

def lcmm(*args):
    """Return lcm of args."""   
    return reduce(lcm, args)

def convert_fracts(fraction_list):
	if len(fraction_list) == []:
		return []
	denom_list = []
	num_list = []
	for pair in fraction_list:
		num, denom = pair
		denom_list.append(denom)
		num_list.append(num)
	GCD = lcmm(*denom_list)
	new_fraction_list = []
	for i, denom in enumerate(denom_list):
		new_num = GCD//denom*num_list[i]
		new_fraction_list.append([new_num, GCD])
	return new_fraction_list

lst = [[1,2], [1,3], [1,4]]

print(f'{convert_fracts(lst)=}')

