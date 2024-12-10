#! /usr/bin/env python3		#Allow Unix shell to execute as a Python script
# _*_ coding: UTF-8 _*_		#Enable unicode encoding
#GMT+0BST-1,M3.5.0/01:00:00,M10.5.0/02:00:00

__author__ = "Ian Pottinger"
__date__ = "20/12/2012"
__contact__ = "ianpottinger@me.com"
__version__ = "1.3.5.7.9 even avoidance"
__credits__ = "Commonly known as Potts"
__copyright__ = "Copyleft for balance"
__license__ = "Whatever Potts Decides"
__metadata__ = [__author__, __date__, __contact__, __version__,
                __credits__, __copyright__, __license__]

import keyword
import multiprocessing
import concurrent
import threading
import asyncio

DEBUG_MODE = True
if DEBUG_MODE:
    import pdb
    #pdb.set_trace()
    import logging
    FORMAT = '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
    logging.basicConfig(level = logging.INFO, format = FORMAT)
    #logging.basicConfig(level = logging.WARNING, format = FORMAT)
    #logging.basicConfig(level = logging.DEBUG, format = FORMAT)
    #logging.basicConfig(level = logging.ERROR, format = FORMAT)
    #logging.basicConfig(level = logging.CRITICAL, format = FORMAT)

RESERVED = ['False', 'None', 'True', 'and', 'as', 'assert', 'break',
            'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'exec',
            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
            'lambda', 'nonlocal', 'not', 'or', 'pass', 'print',
            'raise', 'return', 'try', 'while', 'with', 'yield']
KEYWORDS = keyword.kwlist

import string

def make_magic_square(n):
	if n < 1 or n == 2:
		return False
	if n % 2 == 1:
		return make_odd_magic_square(n)
	else:
		if n % 4 == 0:
			return make_even_4n_magic_square(n)
		elif (n-2) % 4 == 0:
			return make_even_4np2_magic_square(n)
		else:
			return False


def make_panmagic_square(n):
	if n < 4:
		return False
	if n % 2 == 0:
		return make_even_panmagic_square(n)
	else:
		return make_odd_panmagic_square(n)


def make_odd_magic_square(n):
	"""
	Source: http://blogs.mathworks.com/cleve/2012/11/05/magic-squares-part-2-algorithms/
	This method is called the Siam
	Build I and J matrices as follows:

		1 1 1 1 1		1 2 3 4 5
		2 2 2 2 2		1 2 3 4 5
	I=> 3 3 3 3 3	J=> 1 2 3 4 5
		4 4 4 4 4		1 2 3 4 5
		5 5 5 5 5		1 2 3 4 5
	"""
	if n < 1 or n%2 == 0: return False	#only allow odd squares 2n-1, n>0

	J = [range(1, n+1)] * n
	I = transpose(J)
	"""
	Using these indices, we generate two more matrices A and B
		A = mod(I + J + (n-3)/2, n)
		B = mod(I + 2*J - 2, n)

		3 4 0 1 2		1 3 0 2 4
		4 0 1 2 3		2 4 1 3 0
	A=> 0 1 2 3 4	B=> 3 0 2 4 1
		1 2 3 4 0		4 1 3 0 2
		2 3 4 0 1		0 2 4 1 3
	"""
	A = [[(I[i][j] + J[i][j] + (n-3)/2) % n for i in range(n)] for j in range(n)]
	B = [[(I[i][j] + 2*J[i][j] - 2) % n for j in range(n)] for i in range(n)]
	"""
	Both A and B are fledgling magic squares. They have equal row, column and
	diagonal sums. But their elements are not the integers from 1 to n^2. Each
	has duplicated elements between 0 and n-1. The final statement:

		M = n*A + B + 1

	produces a matrix whose elements are integers between 1 and n^2 and which
	has equal row, column and diagonal sums. What is not obvious, is that
	there are no duplicates and includes all of the integers between 1 and n^2
	making it a magic square.
	"""
	return [[n*A[i][j] + B[i][j] + 1 for j in range(n)] for i in range(n)]


def make_even_4np2_magic_square(n):
	"""
	Source: http://www.math.wichita.edu/~richardson/mathematics/
	        magic%20squares/even-ordermagicsquares.html
	The algorithm starts by partitioning the square into 4 blocks of n/2 each;
	one for each quadrant.  We 	fill those blocks with a known odd sized magic
	square, A,  as: A in Q II, A + (n/2)^2 	in Q III, A + 2*(n/2)^2 in Q I, and
	A + 3*(n/2)^2 in Q IV.

	This maintains the proper column totals, but not the row or diagonal totals.
	Correcting row totals is achieved by swaping selective columns (0, 1, 9) of the upper
	10x10 matrix with the lower half.  The columns affected are 0, 1, 2, ..., k
	and n-1, n-2, ... n-k.  (k=(n-2)/4)
	Example (zero based):
		n=6	  columns 0
		n=10  columns 0, 1, 9
		n=14  columns 0, 1, 2, 12, 13
		n=18  columns 0, 1, 2, 3, 15, 16, 17

	The final step is to switch 2 values in certain columns to correct the diagonals
	and make a magic square. (nx-k-1, [k-1, k]) <-> (nx+k, [k-1, k])
	Example (zero based):
		n=6		(1, 0) <-> (4, 0), (1, 1) <-> (4, 1)
		n=10	(2, 1) <-> (7, 1), (2, 2) <-> (7, 2)
		n=14	(3, 2) <-> (10, 2), (3, 3) <-> (10, 3)
		n=18	(4, 3) <-> (13, 3), (4, 4) <-> (13, 4)

	     Fill in 4 Quads with 5x5 M.S.               Swap upper and lower column sections
	17  24   1   8  15  67  74  51  58  65			92  99   1   8  15  67  74  51  58  40
	23   5   7  14  16  73  55  57  64  66			98  80   7  14  16  73  55  57  64  41
	 4   6  13  20  22  54  56  63  70  72			79  81  13  20  22  54  56  63  70  47
	10  12  19  21   3  60  62  69  71  53			85  87  19  21   3  60  62  69  71  28
	11  18  25   2   9  61  68  75  52  59			86  93  25   2   9  61  68  75  52  34
	92  99  76  83  90  42  49  26  33  40			17  24  76  83  90  42  49  26  33  65
	98  80  82  89  91  48  30  32  39  41			23   5  82  89  91  48  30  32  39  66
 	79  81  88  95  97  29  31  38  45  47			 4   6  88  95  97  29  31  38  45  72
	85  87  94  96  78  35  37  44  46  28			10  12  94  96  78  35  37  44  46  53
	86  93 100  77  84  36  43  50  27  34			11  18 100  77  84  36  43  50  27  59
	You will see our typical 5 x 5 magic
	square in Q II (top left corner)

	The finished M.S after the 2 switches
	92  99   1   8  15  67  74  51  58  40
	98  80   7  14  16  73  55  57  64  41
	79   6  88  20  22  54  56  63  70  47
	85  87  19  21   3  60  62  69  71  28
	86  93  25   2   9  61  68  75  52  34
	17  24  76  83  90  42  49  26  33  65
	23   5  82  89  91  48  30  32  39  66
	 4  81  13  95  97  29  31  38  45  72
	10  12  94  96  78  35  37  44  46  53
	11  18 100  77  84  36  43  50  27  59
	"""

	nx, k = n/2, (n-2)/4
	if n < 6 or nx % 2 == 0: return False	#only allow even squares 4n+2, n>1

	#make an odd nx x nx magic square
	A = make_odd_magic_square(nx)

	#fill in each quadrant with an augmentation of A according to algorithm
	I = A + sc_add(A, 3*nx*nx)
	J = sc_add(A, 2*nx*nx) + sc_add(A, nx*nx)

	#create initial square by concatenating I and J - column sums are "magic"
	B = [I[i]+J[i] for i in range(n)]

	#swap upper and lower halves of specific columns to make row sums "magic"
	for j in range(k) + range(n-1, n-k, -1):
		for i in range(nx):
			B[i][j], B[i+nx][j] =  B[i+nx][j], B[i][j]

	#switch middle values for 2 columns to make diagonals (and square) magic
	B[nx-k-1][k-1], B[nx+k][k-1] = B[nx+k][k-1], B[nx-k-1][k-1]
	B[nx-k-1][k], B[nx+k][k] = B[nx+k][k], B[nx-k-1][k]

	return B


def make_even_4n_magic_square(n):
	"""
	Source: http://mathworld.wolfram.com/MagicSquare.html
	An elegant method for constructing magic squares of doubly even order n=4m
	is to draw Xs through each 4 x 4 subsquare and fill all squares in sequence.
	Then replace each entry A[i][j] on a crossed-off diagonal by (n^2+1) - A[i][j]
	or, equivalently, reverse the order of the crossed-out entries.

	Here is a mask of the crossed out matrix:
		1 0 0 1 1 0 0 1
		0 1 1 0 0 1 1 0
		0 1 1 0 0 1 1 0
		1 0 0 1 1 0 0 1
		1 0 0 1 1 0 0 1
		0 1 1 0 0 1 1 0
		0 1 1 0 0 1 1 0
		1 0 0 1 1 0 0 1
	"""
	if n < 4 or n%4: return False	#only allow even squares 4n, n>0

	c, cms, A = 1, n*n + 1, [[0]*n for i in range(n)]
	for i in range(n):
		for j in range(n):
			A[i][j] = cms-c if i%4 == j%4 or (i+j)%4 == (n-1)%4 else c
			c += 1

	return A


def make_even_panmagic_square(n):
	"""
	Description and algorithm from Wikipedia

https://en.wikipedia.org/wiki/Pandiagonal_magic_square

	A pandiagonal magic square remains pandiagonally magic not only under
	rotation or reflection, but also if a row or column is moved from one
	side of the square to the opposite side. As such, an n x n pandiagonal
	magic square can be regarded as having 8n^2 orientations.
	No (4n+2)x(4n+2) panmagic square exists.
	"""
	if n < 4 or n%4: return False	#only allow 4n even squares
	nx = n/2
	x = range(1, n+1)
	"""
	Integrating a few steps into one, compose a matrix as follows:
	(example for n=2 (8x8))
	[1,2,3,4,8,7,6,5]
	[8,7,6,5,1,2,3,4]
			...
			...
	repeat pattern until full, then rotate 90 degrees
	"""
	A = (x[:nx]+x[:nx-1:-1], x[:nx-1:-1]+x[:nx]) * nx
	B = rotate_ccw(A)

	"""
	Build the final square by multiplying the B square by n, adding
	the A square and subtract n from each cell.
	Example: A + n x B - n
	"""
	return [[a + n*b - n for a, b in zip(r1, r2)] for r1, r2 in zip(A, B)]


def make_odd_panmagic_square(n):
	#order 6n +/- 1: 5, 7, 11, 13, 17, 19, 23, ...
	if n > 4 and (n%6 == 1 or n%6 == 5):
		A = [[(j*2 + i) % n + 1 for j in range(n)] for i in range(n)]

	#order 6n + 3: 9, 15, 21, 27, ...
	elif n > 8 and n%6 == 3:
		q, dir = n / 3, 1
		B = [[1,2,3],
			 [5,6,4],
			 [9,7,8]]

		for i in range(10, n, 3):
			B.append(range(i, i+3)[::dir])
			dir = -dir

		A = [[0]*n for i in range(n)]
		for i in range(q):
			for j in range(3):
				t = B[i % q][j % 3]
				for k in range(n):
					A[(i+k) % n][(j + 3*k) % n] = t
	else:
		return False

	T = transpose(A)
	return [[a + n*b - n for a, b in zip(r1, r2)] for r1, r2 in zip(A, T)]


def check_magic_square(matrix, seq=True, mc=0):
	"""
	seq: Boolean used to restrict the square to having distinct numbers from 1..n^2
	mc: Integer to specify a magic constant
	A (normal) magic square is a square array(1) of numbers consisting of the distinct
	positive integers 1, 2, ..., n^2(2) arranged such that the sum of the
	n numbers in any horizontal(3), vertical(4), or main diagonal(5) line is always the
	same number (Kraitchik 1942, p. 142; Andrews 1960, p. 1; Gardner 1961, p. 130;
	Madachy 1979, p. 84; Benson and Jacoby 1981, p. 3; Ball and Coxeter 1987, p. 193),
	known as the magic constant(6)

	A square that fails to be magic only because one or both of the main diagonal sums
	do not equal the magic constant is called a semimagic square(5)

	It is an unsolved problem to determine the number of magic squares of an arbitrary
	order, but the number of distinct magic squares (excluding those obtained
	by rotation and reflection) of order n=1, 2, 3, ... are 1, 0, 1, 880, 275305224, ...
	(Sloane's A006052; Madachy 1979, p. 87).(6)
	"""
	global err_message

	#(6) check for invalid size and shape (non-square)
	n = square_size(matrix)
	if (n < 1 or n == 2):
		err_message = "The array must be square and can't be <1 or 2."
		return False

	#(1) quick check for a square array
	if (not isinstance(matrix[0], (list, tuple)) or
						any(len(row) != n for row in matrix)):
		err_message = "Must be a square, i.e. the same number of rows and columns."
		return False

	#(6) calculate the magic constant and working variables
	if not mc: mc = n * (n*n + 1) / 2
	d1, d2, r, c, a = 0, 0, [0]*n, [0]*n, set()
	for i in range(n):
		d1, d2 = d1+matrix[i][i], d2+matrix[i][n-i-1]
		for j in range(n):
			r[i] += matrix[i][j]
			c[j] += matrix[i][j]
			a.add(matrix[i][j])

	#(2) check distinct positive integers 1, 2, ..., n^2
	if seq == True and not all(x in a for x in range(1, n*n+1)):
		err_message = "The numbers 1 through " + str(n*n) + \
						" must appear once and only once in the array."
		return False

	#(3) check that all horizontal lines equal the magic constant
	row_sum = set(r)
	if (len(row_sum) != 1 or mc not in row_sum):
		err_message = "All the rows didn't add to the magic constant, " + str(mc)
		return False

	#(4) check that all vertical lines equal the magic constant
	col_sum = set(c)
	if (len(col_sum) != 1 or mc not in col_sum):
		err_message = "All the columns didn't add to the magic constant, " + str(mc)
		return False

	#(5) check that both diagonal lines equal the magic constant
	if (d1 != mc or d2 != mc):
		err_message = "One or both diagonals didn't add to the magic constant, " \
						+ str(mc) + "\nThis is a semimagic square."
		return False

	return True


def check_panmagic_square(A):
	"""
	If all diagonals (including those obtained by wrapping around) of a magic
	square sum to the magic constant, the square is said to be a panmagic
	square (also called a diabolic square or pandiagonal square). There are no
	panmagic squares of order 3 or 4n+2.
	"""
	n = square_size(A)
	if n < 1 or n == 3 or (n-2)%4 == 0: return False

	if check_magic_square(A):	#make sure A is magic before panmagic test
		mc = n * (n*n + 1) / 2	#magic constant

		for i in range(n):
			s1 = sum(A[(i-j) % n][j] for j in range(n))
			s2 = sum(A[(i+j) % n][j] for j in range(n))
			if s1 != mc or s2 != mc: return False
		return True

	return False


def check_associative_square(A):
	"""
	Reference: https://en.wikipedia.org/wiki/Associative_magic_square
	An associative magic square is a magic square for which every pair
	of numbers symmetrically opposite to the center sum up to the same
	value.  There are no associative magic squares of singly-even order.
	"""
	n = square_size(A)
	if n < 1 or n%4 != 0 and n%2 == 0: return False

	if check_magic_square(A):	#make sure A is magic before associative test
		amc = n*n + 1	#associative magic constant
		odd_square = n % 2
		if all(A[i][j] + A[n-i-1][n-j-1] == amc
				for i in range(n) for j in range(n/2 + odd_square)):
			return True

	return False


def check_bimagic_square(A):
	"""
	If replacing each number in A with its square produces another magic
	square, then the square is said to be a bimagic square (or doubly
	magic square).
	"""

	if check_magic_square(A):	#make sure A is magic before bimagic test
		n = len(A)
		np = n*n
		mc = n * (np + 1) * (2*np + 1) / 6	#magic constant

		if check_magic_square([[x*x for x in row] for row in A], False, mc):
			return True

	return False


"""
Basic matrix operations in native Python.  Most of the time we simulate
these operations by accessing the matrix in a different order.
"""
def square_size(A):
	try:
		n = len(A)
	except:
		return 0

	if (not isinstance(A[0], (list, tuple)) or
			any(len(row) != n for row in A)):
		return 0
	return n


def transpose(A):
	"""
	reflect A over its main diagonal (which runs from top-left to bottom-right)
	Example:
		1, 2, 3		1, 4, 7
		4, 5, 6  => 2, 5, 8
		7, 8, 9		3, 6, 9

	"""
	return [list(a) for a in zip(*A)]


def rotate_cw(A):
	"""
	rotatet A clockwise (right-turn, -90 degrees)
	Example:
		1, 2, 3		7, 4, 1
		4, 5, 6  => 8, 5, 2
		7, 8, 9		9, 6, 3

	"""
	return [list(a) for a in zip(*A[::-1])]


def rotate_ccw(A):
	"""
	rotate A counterclockwise (left-turn, 90 degrees)
	Example:
		1, 2, 3		3, 6, 9
		4, 5, 6  => 2, 5, 8
		7, 8, 9		1, 4, 7

	"""
	return [list(a) for a in zip(*A)[::-1]]


def flip(matrix):
	"""
	flip A over horizontally
	Example:
		1, 2, 3		7, 8, 9
		4, 5, 6  => 4, 5, 6
		7, 8, 9		1, 2, 3

	"""
	return matrix[::-1]


def sc_add(A, n):
	"""
	Scalar add n to matrix A
	"""
	return [[x+n for x in row] for row in A]


def print_matrix(matrix):
	"""
	Source: David Robinson from

http://stackoverflow.com/questions/8747500/pythons-pretty-printing-of-matrix

	"""
	max_lens = [max([len(str(r[i])) for r in matrix])
                for i in range(len(matrix[0]))]

	print "\n".join(["".join([string.rjust(str(e), l + 2)
                for e, l in zip(r, max_lens)]) for r in matrix])


def demo(A):
	if check_magic_square(A):
		print_matrix(A)
		print "this is a magic square"
	else:
		print err_message
		return False
	if check_panmagic_square(A): print "**panmagic"
	if check_associative_square(A):	print "**associative"
	if check_bimagic_square(A):	print "**bimagic"
	print

def swap_col(A, n, m):
	l = len(A)
	for i in range(l):
		A[i][n], A[i][m] = A[i][m], A[i][n]
	return A

def swap_row(A, n, m):
	l = len(A)
	for i in range(l):
		A[n][i], A[m][i] = A[m][i], A[n][i]
	return A


"""
n = 4*1
print "4n (doubly even) pandiagonal or panmagic square, n=",n/4
M = make_even_panmagic_square(n)
for x in range(1):
	demo(M)
	N = rotate_cw(M)
	demo(rotate_cw(M))
	demo(rotate_cw(rotate_cw(M)))
	demo(rotate_ccw(M))
	demo(flip(M))
	demo(flip(rotate_ccw(M)))
	demo(flip(rotate_cw(M)))
	demo(flip(rotate_cw(rotate_cw(M))))
	M=swap_row(N, 0,1)
	M=swap_row(M, 2,3)

"""

n = 7
print "Odd magic square, n=",n
M = make_magic_square(n)
demo(M)

print "Odd panmagic square, 6n+1, 6n+5"
n = 7
M=make_panmagic_square(n)
demo(M)

n = 11
M=make_panmagic_square(n)
demo(M)

print "Odd panmagic square, 6n+3"
n = 9
M=make_panmagic_square(n)
demo(M)

n = 4*2
print "4n (doubly even),  magic square, n=",n/4
M = make_magic_square(n)
demo(M)

n = 4*2
print "4n (doubly even) panmagic square, n=",n/4
M = make_panmagic_square(n)
demo(M)

print "8 x 8 bimagic square test check"
T =[[56, 34,  8, 57, 18, 47,  9, 31],
	[33, 20, 54, 48,  7, 29, 59, 10],
	[26, 43, 13, 23, 64, 38,  4, 49],
	[19,  5, 35, 30, 53, 12, 46, 60],
	[15, 25, 63,  2, 41, 24, 50, 40],
	[ 6, 55, 17, 11, 36, 58, 32, 45],
	[61, 16, 42, 52, 27,  1, 39, 22],
	[44, 62, 28, 37, 14, 51, 21,  3]]
demo(T)

n=4*2 + 2
print "Even, 4n + 2, magic square, n=",(n-2)/4
M = make_magic_square(n)
demo(M)

print "Even, 4n, square associative test"
T = [[2,13,16,3],
	 [11,8,5,10],
	 [7,12,9,6],
	 [14,1,4,15]]
demo(T)

print "Odd square associative test"
T = [[25,22,5,10,3],
	 [7,12,11,17,18],
	 [2,6,13,20,24],
	 [8,9,15,14,19],
	 [23,16,21,4,1]]
demo(T)
