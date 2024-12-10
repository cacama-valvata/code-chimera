

### None ###
# None is falsy, meaning that it evaluates to False in an if statement;
# None is also unique, which means that you can test if something is None using this keyword.

z_man = None
if z_man:  # evaluaes to false
    print("z man exists!")

if z_man is None:

    ### Identity and Equality of data ###
    # https://stackoverflow.com/questions/132988/is-there-a-difference-between-and-is
    # is and ==;
    # is tests for if it is the same object
    # == tests for equality of value
    # is will return True if two variables point to the same object, == if the objects referred to by the variables are equal.

a = [1, 2, 3]
b = a[:]

if a is b:
    print("a is b!")  # does not print
if a == b:
    print("a == b!")

c = 35
d = 35
if c is d:
    print("c is d!")
if c == d:
    print("c == d!")
