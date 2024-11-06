#LC	 	=	 	num
# 	 	|	 	id
# 	 	|	 	(λ (id) LC)
# 	 	|	 	(LC LC)
# 	 	|	 	(+ LC LC)
# 	 	|	 	(* LC LC)
# 	 	|	 	(ifleq0 LC LC LC)
# 	 	|	 	(println LC)

import sys;
import re;
#Production Rules
#num
def isNum(num):
	return True

#http://stackoverflow.com/questions/42070323/split-on-spaces-not-inside-parentheses-in-python for this elegent peice of code
def split_parens(expr):
	numBrackets = 0;
	l = [0];
	expr = expr.strip(" ");
	for i, c in enumerate(expr):
		
		if c == "(":
			numBrackets+=1
		elif c == ")":
			numBrackets-=1
		elif c == " " and numBrackets == 0:
			l.append(i)
		if numBrackets < 0:
			raise Exception("Syntax error, too many closing parens.")

	l.append(len(expr))

	if numBrackets > 0:
			raise Exception("Syntax error, not enough parens")
	#combine strings  by combining sets of ranges from the funciont
	return([expr[i:j].strip(" ") for i,j in zip(l,l[1:])])


def compile_primative(exprList, bindings):
	return str(exprList[0])

def compile_lambda(exprList, bindings):

	extractedId = exprList[1].strip("()")
	#newBindings = bindings.append(extractedId)
	rest = exprList[2]
	return "(lambda " + extractedId  + " : " + compile_expr(rest,bindings)+" )"

def compile_application(exprList, bindings):
	aExpr = exprList[0];
	bExpr = exprList[1];
	return compile_expr(aExpr,bindings) + " (" + compile_expr(bExpr,bindings) + ")"

def compile_addition(exprList, bindings):
	aExpr = exprList[1];
	bExpr = exprList[2];
	return compile_expr(aExpr,bindings) + " + " + compile_expr(bExpr,bindings) 

def compile_multiplication(exprList, bindings):
	aExpr = exprList[1];
	bExpr = exprList[2];
	return compile_expr(aExpr,bindings) + " * " + compile_expr(bExpr,bindings) 

def compile_ifleq0(exprList, bindings):
	#Use Termary operator  for issues w/ tabs
	exprTest  = exprList[1]
	exprTrue  = exprList[2]
	exprFalse = exprList[3]
	return ("("+compile_expr(exprTrue,bindings) + ") if (" + compile_expr(exprTest,bindings) + ") <= 0 else (" + compile_expr(exprFalse,bindings) +")")

def compile_print(exprList, bindings):
	return "(lambda : print (" + compile_expr(exprList[1],bindings) + "))()"

def compile_expr(expr, bindings):
	expr = expr.strip(" ");
	if expr.startswith('(') and expr.endswith(')'):
   		expr = expr[1:-1]
	exprList = split_parens(expr)

	
	if len(exprList) == 1:
		return compile_primative(exprList,bindings)
	elif len(exprList) == 2:
		sym = exprList[0]
		if sym == "println":
			return compile_print(exprList, bindings)
		else:
			return compile_application(exprList,bindings)
	elif len(exprList) >= 3:
		sym = exprList[0]
		if sym == "λ":
			return compile_lambda(exprList,bindings)
		elif sym == "+":
			return compile_addition(exprList,bindings)
		elif sym == "*":
			return compile_multiplication(exprList,bindings)
		elif sym == "ifleq0":
			return compile_ifleq0(exprList,bindings)



def top_compile_expr(expr):
	expr = expr.replace("\n"," ");
	expr = expr.replace("\r"," ");
	expr = re.sub(' +', ' ' , expr);
	a = []
	return compile_expr(expr,a)

f = open(sys.argv[1])
out = open(sys.argv[1] + ".py","w")
a = top_compile_expr(f.read())
out.write("import sys\nsys.setrecursionlimit(100000)\n")
out.write("print(" + a + " )")

