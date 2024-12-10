# Question 1: Make sure python is loaded and working
# Uncomment the below code and run it
# print("Hello world!")
# TODO: BETHANY - HAVE PEOPLE PRINT OUT THEIR VERSION

#TODO: ALEX - have some memory sticks of tutorial

#ALEX
# Question 2: Get rid of all of the duplicates in this list.
# Think about the complexity of your solution.
# Complexity cheetsheet http://bigocheatsheet.com/
# You will find it is O(n^2), please make sure you can explain why as an exercise
# Cool link https://www.peterbe.com/plog/uniqifiers-benchmark
# https://wiki.python.org/moin/TimeComplexity

#myList = ['c', 'd', 'c', 'a', 'b', 'c', 'a', 'z', 'd', 'e', 'f', 'f', 'g']


# Answer Question 2
#newList = []
#for i in myList:
#    if i not in newList:
#        newList.append(i)

#print(newList)
###################
#Pause Here for a Word 'Pythonic' and Introduction to Comprehensions
###################

'''
______________________________________
"PYTHONIC"
______________________________________

Thank you Python Foundation & Lea Verou

PEP8 (Code Formatting)            https://www.python.org/dev/peps/pep-0008/
An alternate view on spaces       http://lea.verou.me/2012/01/why-tabs-are-clearly-superior/

PEP202 (list comprehensions)      https://www.python.org/dev/peps/pep-0202/
PEP274(dictionary comprehensions) https://www.python.org/dev/peps/pep-0274/
PEP20 (The Zen of Python)         'import this' in the Python shell

"Programs must be written for people to read, 
and only incidentally for machines to execute." -- Harold Abelson (Structure and Interpretation of Computer Programs)

"Code is read much more often than it is written" -- Guido VanRossum


--many formatting rules and conventions, all centered around making code scanable
and easily ready by humans.  In addition, Python cares about indentation (as we know!) - but only 
about *indentation level relative to other levels* so you can indent 2, 3, 4, [tab] -- whatever you want.
The best advice is to pick an indentation distance, and stick to it.
Biggest no-no?  Don't mix tabs with spaces.  Ugly.


Some other quick takeaways:

1)  Code should be written in an implementation agnostic way whenever possible.
	Keep in mind that Python has PyPy, Jython, IronPython, Cython, Psyco, and other implementations.
    
    example:  string cocatenation using '+' -- this is only (somewhat) efficient in CPython
   	-- but doesn't even exist as a possiblilty in some implementations.  Using ''.join() 
   	is the safest and most performant way of writing concatenation, unless you know
   	that the implementation will always be CPython.

2)  Comparisons to singletons like None should be done with 'is' or not -- never
    the equality operators.

3)  Use not rather than not...is

	YES:  if foo is not None
	NO:   if not foo is None

4)  Always use a def statement instead of an assignment statment that binds a 
    lambda expression directly to an identifier.

    YES:  def f(x):  return 2*x
    NO:   f= lamda x: 2*x


So really -- what's Pythonic?  Readable, understandable, clean, and concise.
(and -- as a bonus, as performant as the CPython implementers can make it).


How does this apply to list comprehensions??

	1)  They are concise -- an entire block of code can be compressed into one line 
	    (although this can be abused).
	
	2)  Some consider them more easily read (less lines of code to scan, more logical order to the operations)
	
	3)  They *generally* produce fewer function calls in Python than the map/reduce/filter alternatives.
	    In Python, function calls are **expensive** because they are added to the stack.

	4)  They are tuned in Python to be performant (although this varies by application.)  
	    See http://python-history.blogspot.com/2010/06/from-list-comprehensions-to-generator.html
        for gorey gorey details and http://leadsift.com/loop-map-list-comprehension/ for some interesting timings)

_______________________________________________________
LIST COMPREHENSIONS
________________________________________________________
Thank you
Bruce Eckel  -- Python3 Patterns & Idioms   http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Comprehensions.html
Mark Pilgrim -- Dive into Python3           www.diveintopython3.net
Obi Ike-Nwosu                               http://intermediatepythonista.com/python-comprehensions
Python Foundation                           https://docs.python.org/3/howto/functional.html.  

With exercises help from  
Trey Hunner                                 http://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/
Boston Python                               http://puzzles.bostonpython.com/
github/zhiwehu/Python-programming-exercises


Quick Examples

this										  becomes

symbols = '$¢£¥€¤'                            symbols = '$¢£¥€¤'
codes = []                                    codes = [ord(caracter) for character in symbols]
for symbol in symbos:                          
	codes.append(ord(symbol))



this        								  becomes
							  
results = []								  results = [num**2 if num%2==0 else num**3 for num in range(1, 21)]

for num in range(1, 21):
	if num%2 == 0:
		results.append(num**2)
	else: 
		results.append(num**3)



So WTF are comprehensions?  
Comprehensions in Python are syntactic constructs that allow sequences
to be built from other sequences. Python3 (and Python 2.7.0++) come 
with different comprehension flavors:  List, Dictionary, and Sets.

A comprehension generally has 4 parts:

1)  An input sequence (in Python, this is anything that's *iterable*)
	thise includes strings, files, lists, arrays, sets, streams, or dictionaries
	see https://docs.python.org/3.5/glossary.html#iterable for more info

2)  One (or more) variables representing the members of the input sequence

3)  An optional predicate expression used to filter the data for output

4)  An output expression used to produce elements of the output data 
    structure from members of the input sequence


The easiest way to think about it is as a compacted, flattened loop, although it's a 
little bit of a shift in thinking:


[Output Expression(4)  Input Sequence(2&1)  Optional Predicate (3)]

input_list = [1,2,3,4,5,6,7,8,9]
output_list = []

for item in input_list:              -->  this is the Input Expression
	if item%2 == 0:                  -->  this is the Predicate, or *Filter*
		output_list.append(item**2)  -->  this is the Output Expression


turns into

input_list = [1,2,3,4,5,6,7,8,9]
output_list = [item**2 for item in input_list if item%2==0]


Of course, not everything can be (or should be!) compressed in this way.



Rules of thumb for comprehensions:
1)  Use a comprehension if you know your outpt needs to be in list 
    (or dictionary or set) and your input(s) is of an *iterable* type

2)  Use a comprehension if you find yourself looping through a set of expressions 
    to produce values that are appended to a list, dictionary or set 
    (If iterations are executed in order to build a composite value)

3)  Use a comprehension if a loop increases the number of function calls
    (.append() is a function call that's omitted if you use a comprehension)

4)  If you are only using the list for it's *side effects*, consider using another
    format (lists are compairatively memory expensive)
    ...ie reversing a sentance by reverse sorting and re-joining.

5)  If you have to nest several comprehensions, consider seperating out the steps
    or the loop to increase readability (readability counts!)
'''
#______________________________________________
#COMPREHENSION EXERCISES
#---------------------------------------------------------------------------------------------------------------------
#Question 3
#Convert the first exercise into a comprehension
'''
my_list = ['c', 'd', 'c', 'a', 'b', 'c', 'a', 'z', 'd', 'e', 'f', 'f', 'g']
new_list = []

For i in myList:
    if i not in newList:
        newList.append(i)

print(new_list)


Solution:
new_list = [x for x in my_list if some_list.count(x) > 1]
print(new_list)
'''
_______________________________________________________________
#Question 4
#Convert this list and loop into a list comprehension

'''
input_list = [25, 8, 'C', 'l', 'Z', '7', 'l', 'g', 'u', 19, 14, 7, '7', 'o', 3, 17, 6, 21, 'q', 21, 'T', 6, 23, 'M', 'B', 9, 4, 22, 'w', 20, 'D', 'w', 'D', 7, '0', '7', 8, 'Z', 1, 18, 4, 'Q', 'W', 15, 'K', 13, 8, 'k', 0, 11]
squared_numbers = []

for item in input_list:
	if isinstance(item, int):
		squared_numbers.append(item**2)


Solution:

squared_numbers = [number**2 for number in my_list if isinstance(number, int)]
'''
________________________________________________________________
#Question 5
#Convert this list/list comprehension into a list/loop:
'''
input_list = ['s', 'dd', 'v', 'TT', 'l', 'r', 'II', 'u', 'H', 'E', 'mm', 'qq', 'o', 'UU', 'nn', 'RR', 'YY', 'c', 'BB', 'P', 'ff', 'f', 'nn', 'bb', 'D', 'JJ', 'hh', 'k', 'y', 'F', 'J', 'dd', 'kk', 'JJ', 'L', 'jj', 'cc', 'f', 'P', 'KK', 'L', 'vv', 'CC', 'D', 'z', 'SS', 'L', 'E', 'r', 'b']

doublecaps = [item for item in input_list if item.isupper() and len(item)>1]


Solution:

doublecaps = []
for item in input_list:
	if item.isupper() and len(item)> 1:
		doublecaps.append(item)
'''
_____________________________________________________

#Question 6
#Convert This list and loop into a list comprehension
'''
import string

input_list = ['w', 8, 12, 4, 13, 'O', '7', 20, 'q', '3', 'v', 'E', '4', 1, 18, 1, 'e', 'I', 23, 'n', 12, 8, 3, 5, 5, 3, 21, 'H', 19, 14, 5, 'a', 'Z', 'Y', 23, 'g', 'p', 'Y', 'r', 'j', 'y', 'x', 0, '0', 16, 'U', 'S', 'k', 'D', 'D']

numbers_and_letters = []

for item in input_list:
	if isinstance(item, int):
		numbers_and_letters.append(item**2)
	elsif item.isalpha():
		numbers_and_letters.append(item)


Solution:
import string
numbers_and_letters = [item**2 if isinstance(item, int) else item if item.isalpha() else None for item in my_list]
'''
______________________________________________________
#Question 7

#Write a program which will find all numbers between 2000 and 3200 which are divisible by 7 but are not a multiple of 5.
#Output should be a list (try doing this with a comprehension)

'''
Solution:
l=[]
for i in range(2000, 3201):
    if (i%7==0) and (i%5!=0):
        l.append(i)

OR

l = [i for i in range(2000, 3201) if (i%7==0) and (i%5!=0)]
'''

___________________________________________
#Question 8
#Write a function/comprehension which can compute the factorials of a list of numbers.
#The results should be returned in a list.

#Suppose the following input is supplied to the program:
#[7, 8, 9]

#Then, the output should be:
#[5040, 40320, 362880]

#Hint:  You can use the math module (math.factorial()) or write your own helper function

#Extras: Do it without calling the math module (Hint: try using reduce()), try using reduce() and operator.mul, try using only nested lists inside the comprehension.


'''
input_list = [7,8,9]


Solution #1 using the math module:

from math import factorial
solution = [factorial(x) for x in input_list]


Solution #2 using helper functions:

def fact(x): #recursive helper method
    if x == 0:
        return 1
    return x * fact(x - 1)


def fact(x):  #iterative helper method
	factorial = 1
	
	for number in range(1, x+1):
		factorial = factorial*number
	return factorial


solution = [fact(x) for x in input]
 


Extra Method #1 using a lambda:

from functools import reduce
solution = [reduce(lambda a, b: a*b, range(1, number+1)) for number in my_list]

Extra Methon #2 using reduce() and operator.mul

from functools import reduce
from operator import mul
solution = [reduce(mul, range(1, number+1)) for number in my_list]


Extra Method #3 using list slices #do NOT do this at home!!:

solution = [j for j in [1] for i in range(2, fac+1)for j in [j*i]][-1] for number in my_list]
'''

______________________________________

#TIMEIT HERE???

#15 to 20 mins
#TODO : Jouella - PYTHON 3 FUNCTION CALLS ON STACK MEMORY  WRITING EFFECITENT LIST COMPREHENSIONS
#EXERCISES - here
#TODO: Show how tuple is immutable

#_________________________________________
#ALEX
#Question 3 - Create a list of the months of a year 'jan', 'feb', .... ' dec'
# and use the tuple(myList) function to create a tuple of the list
# Mess around with the list methods, count, index, insert, pop, remove, reverse, append and sort
# Notice any differences between tuples and lists

# Background: You can change a list. You cannot change a tuple.
# Another way to say that is lists are mutable and tuples are immutable.
#
# Create a list and create a tuple both that represent the months of a year

#Answer 3
#myListMonths = ['January', 'Feb', 'march', 'april', 'may', 'june', 'july', 'august', 'sept', 'oct', 'nov', 'dec', 'January']
#myTupleMonths = tuple(myListMonths)

#count
# lists
#count = myListMonths.count('January')
#print(count)
#Tuples
#count = myTupleMonths.count('January')
#print(count)

#index








#TODO: ALEX - clean up, Alex to talk about tuples introduce that and lists of tuples why it is helpful
#TUPLES - unique and difference between tuples and lists
#Exercise - Unpacking a sequence into separate variables
# This section is from the python cookbook, 3rd edition
# You can unpack any sequence into variables using a simple assignment operation. The only requirement is that
# the number of variables and structure match the sequence.
# Example
#p = (4,5) # a tuple
#x, y = p
#print(x)
#print(y)

#data = ['ACME', 50, 91.1, (2012, 12, 21)]
#names, shares, price, date = data
#print(date)



#TUPLES ARE HASHABLE - gotcha
# Exercise 3: Build an address book only using lists (the lists can be lists of tuples of objects)

# Answer
# Lists of tuples
# phonebook = [
#    ("John Doe", "555-555-5555"),
#    ("Albert Einstein", "212-555-5555"),
# ]
# print(phonebook)

# phonebookNames = ['John Doe', 'Albert Einstein']
# phonebookNumbers = ['555-555-5555', '212-555-5555']
# print(phonebookNames[0])
# print(phonebookNumbers[0])

#todo alex
# Exercise 4 : Build a look up function for your address book

# Answer
# phonebook = [
#    ("John Doe", "555-555-5555"),
#    ("Albert Einstein", "212-555-5555"),
# ]
# print(phonebook)

# def find_phonenumber(phonebook, name):
#    for n, p in phonebook:
#        if n == name:
#            return p
#        return None

# print "John Doe's phone number is", find_phonenumber(phonebook, "John Doe")

# TODO ALEX: CLEAN THIS UP
# Exercise 5 : Build a phone book using dictionary

# USES

#a = { 'x':1, 'y':2, 'z':3 }
#b = { 'w': 10, 'x': 11, 'y': 12 }


# answer
# phonebook = {
#    "John Doe": "555-555-5555",
#    "Albert Einstein" : "212-555-5555",
# }
# print "John Doe's phone number is", phonebook["John Doe"]

#Vitamins and minerals that go with vegtables
# Shoes and sizes of shoes but you want to track the type of shoe



#ALEX SECTION
# SECTION ON BEGINNING DICTIONARY STUFF - Python Pocket reference section
# Fluent python here the beginning section

#Section that plays with Dictionaries

#Section on initialization - take from O'Reilly Python Pocket Reference by Mark Lutz

#Any immutable object can be a dictionary key (string, number, tuple)
#Class instances can be keys if they inherit hashing methods

# A two item dictionary: keys 'spam' and 'eggs'
#A = {'spam': 2, 'eggs':3}

#Nested dictionaries
#B = { 'info': {42: 1, type(''):2, 'spam':[]}}

#Creates a dictionary by passing keyword arguments to the type constructor
#C = dict(name='Bob', age=45, job=('mgr', 'dev'))

#Dictionary comprehension expression
#D = {c.upper(): ord(c) for c in 'spam'}
#print(D)
#print(ord('s'))

#E = {} # an empty dictionary

#Exercises for this section
#STEP 1 - Print out all of the keys for dictionary A
#print(A.keys())

#STEP 2 - Print out all of the values for dictionary D
#print(D.values())

#STEP 3 - play with the C.items() function.
#print(C.items())

#Step 4 - How would you clear all of the items from dictionary A?
#A.clear()
#print(A)

#Step 5 - Put the values back in A and do a copy (shallow copy) then change a value and print out both
#F = A.copy()
#print(F)
#F['spam'] = 4
#print(A)
#print(F)

#Step 6 - iterate through the dictionary D
#http://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops-in-python
#for key, value in D.items():
#    print key, 'corresponds to', D[key]

#--------------------------------------------------------------------------------------------------------------------
# TO DO : BETHANY - WRITE exercises on this
#BETHNAY - WORK IN SETTING DEFAULT FOR DICTIONARY
# DICTIONARY COMPHREHENSIONS
#END BETHANY SECTION


# TODO BETHANY - Why is it that the dictionary works better
# Discussion  :
# Create a function that times the performance of the list-bisect method
# versus a dictionary for finding a number in a phone book.
# How does the timing scale as the size of the phone book grows?
# https://docs.python.org/3.5/library/timeit.html
# SECTION FOR TIMEIT - BETHANY
#BETHANY  - finding stuff in a list, finding stuff in a dictionary, adding stuff to a list and adding stuff to a dict
# using timeit.
#---------------------------------------------------------------------------------------------------------------------




#TODO ALEX
# Concept of a hash table
# Concept of open addressing (one object per bucket) and chaining (linked list in each bucket)

# Lookup for a dictionary is constant because O(n) lookup and O(1) for insertion

# A unique object that can
# https://wiki.python.org/moin/TimeComplexity



#TODO ALEX
# Excercise 7 - very simple hash table for illustration
# KEY AND VALUE
# SIMPLE HASH TABLE HERE

# Example of something that is unhashable - another list. lists are mutable. if a datatype is mutable it is not hashable
# Because you can't reduce it to a unique value because it might change and the hash funciton would change.


#TODO ALEX  - fluent python book
# Exercise 8 - Illustrate the difference between a hash table and a set. (review dictionary )
# Sets are just hashtables without values or dictionaries without values. Lists are resizable arrays that track
# What can go in a list can be unhashable

# excecise X - Dedup the list from the beginning using a set

# myList = ['c', 'd', 'c', 'a', 'b', 'c', 'a', 'z', 'd', 'e', 'f', 'f', 'g']
# mySet = set(myList) #Because it is using a hash funciton, this is O(n)
# print(mySet)


# Excercise X
# myList = ['c', 'd', 'c', 'a', 'b', 'c', 'a', 'z', 'd', 'e', 'f', 'f', 'g']
# myOtherList = ['c', 'c', 'a', 'z', 'p', 'q', ]
# myListIsNowASet = set(myList)
# myOtherListIsNowASet = set(myOtherList)
# intersaction = myListIsNowASet&myOtherListIsNowASet
# print(intersaction)
# https://docs.python.org/2/library/stdtypes.html#set
#---------------------------------------------------------------------------------------------------------------------

#TODO BETHANY - SET COMPREHENSIONS
# Set Comprehensions - fluent programming book
# Comphrehension of a set
#some_list = ['a', 'b', 'c', 'b', 'd', 'm', 'n', 'n']
#duplicates = set(x for x in some_list if some_list.count(x) > 1)
#print(duplicates)

# SECOND COMPHREHENSION EXAMPLE - TODO: BETHANY Super fast example dedupping with list and a set

#TODO BETHANY - set logical examples
#1.9 Finding Commonalities in Two Dictionaries chapter 1 Data Structures and Algorithms 3rd edition Python cookbook

#____________________________________________________________________________________________________________________

#TODO ALEX
# ABUSES Of Dictionaries

#Adding items to a dict may change the order of existing keys

#MEMORY - don't put too much in memory

#AFTER
#some_list = ['a', 'b', 'c', 'b', 'd', 'm', 'n', 'n']
#deduped = []jojofabe@gmail.com

#for value in some_list:
#    if value not in deduped:
#        deduped.append(some_list.pop(some_list.index(value)))

#print(deduped)
#print(some_list)

#read the dict from start to finish and collect the needed additions in a second dict. Then update the first one with it.
#Exercise to


#A list of hashable items the keys are a list of hashable items. (LIST OR SET)




# - DICS ARE NOT SPACE EFFECIENT - A dictionary is in memory
# For example, if you are handling a large quantity of records,
# it makes sense to store them in a list of tuples or named tuples instead of using a list of dictionaries i
# n JSON style, with one dict per record. Replacing dicts with tuples reduces the memory usage in two ways:
# by removing the overhead of one hash table per record and by not storing the field names again with each record.


# Adding items to a dict may change the order of existing keys

# Set elements must be hashable objects.
#
#



---------------------------------------------------------------------------------------------------------------------

#TODO - JOUELLA PEP8 WHAT IS PYTHONIC - reserve indexiging with slicing and performance and other PEP 8 stuff
#TIMEIT WILL HAVE BEEN INTRODUCED





-
--------------------------------------------------------------------------------------------------------------------
#TODO BETHANY - Peppering stuff there
# Counter, and default dict #DEQUE - double ended queue - as a stack or a queue (append to either end in constant time)
# Ordered dictionaries
# Fluent python here Variations of dict - COLLECTIONS stuff


-------------------------------------------------------------------------------------------------------------------
#TODO - JOUELLA - NAMED TUPLES


#________________________________________________________________________________________________________________

#TODO BETHANY - END SECTION 
#END EXAMPLES - for people finished
#FIND THE MOST Efficient solution for debbing a large list. I want a list of the duplicates and a debbed list and i don't
#want duplicates in the dubbed list

#MARKOV CHAINS and dictionaries
#http://agiliq.com/blog/2009/06/generating-pseudo-random-text-with-markov-chains-u/
