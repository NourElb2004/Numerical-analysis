import sympy as sp
import math
from sympy import symbols, Eq, solve, sympify
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from ttkthemes import ThemedTk

def equation(equ):
    #equation in terms of symbol
    x = sp.Symbol('x')
    #convert string into sympy expression
    equation = sp.sympify(equ)
    #callable function to evaluate symbolic expression numerically
    equation_func = sp.lambdify(x, equation)
    return equation_func
def check_decimal_after_dot(number):
    return str(number).endswith('.0')

def test_coefficients(equ1,equ2,equ3):
    flag=True
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    #take expression before =
    a= equ1.split('=')[0]
    b= equ2.split('=')[0]
    c= equ3.split('=')[0]
    equation1 = sp.sympify(a)
    equation2 = sp.sympify(b)
    equation3 = sp.sympify(c)
    #take coeff before each variable in expression
    equ1coeffs_x1 = equation1.coeff(x1)
    equ1coeffs_x2 = equation1.coeff(x2)
    equ1coeffs_x3 = equation1.coeff(x3)
    equ2coeffs_x1 = equation2.coeff(x1)
    equ2coeffs_x2 = equation2.coeff(x2)
    equ2coeffs_x3 = equation2.coeff(x3)
    equ3coeffs_x1 = equation3.coeff(x1)
    equ3coeffs_x2 = equation3.coeff(x2)
    equ3coeffs_x3 = equation3.coeff(x3)
    #a1=equ1coeffs_x2+equ1coeffs_x3
    #b1=equ2coeffs_x1+equ2coeffs_x3
    #c1=equ3coeffs_x2+equ3coeffs_x1
    #print(a1,b1,c1,equ1coeffs_x1,equ2coeffs_x2,equ3coeffs_x3)
    #test for jacobi 
    if(equ1coeffs_x1>(equ1coeffs_x2+equ1coeffs_x3) and equ2coeffs_x2>(equ2coeffs_x1+equ2coeffs_x3) and equ3coeffs_x3>(equ3coeffs_x2+equ3coeffs_x1)):
        flag=False
    return flag

def isolatex1(equ):
    #x1 in one side equation
    x1, x2, x3 = symbols('x1 x2 x3')
    #split two equations before and after =
    left_side, right_side = equ.split('=')
    #convert string into sympy expression
    equation = Eq(sympify(left_side) - sympify(right_side), 0)
    #solve equation in terms of x1=
    isolated_x1 = solve(equation, x1)[0]
    print(f"x1 = {isolated_x1}")
    return isolated_x1
    
def isolatex2(equ):
    x1, x2, x3 = symbols('x1 x2 x3')
    left_side, right_side = equ.split('=')
    equation = Eq(sympify(left_side) - sympify(right_side), 0)
    isolated_x2 = solve(equation, x2)[0]
    print(f"x2 = {isolated_x2}")
    return isolated_x2
    
def isolatex3(equ):
    x1, x2, x3 = symbols('x1 x2 x3')
    left_side, right_side = equ.split('=')
    equation = Eq(sympify(left_side) - sympify(right_side), 0)
    isolated_x3 = solve(equation, x3)[0]
    print(f"x3 = {isolated_x3}")
    return isolated_x3
    
def equationx3(equ):
    #convert to expression after each variable in one side in terms of other variables
    x1 = sp.Symbol('x1')
    x2 = sp.Symbol('x2')
    equation = sp.sympify(equ)
    equation_func = sp.lambdify((x1, x2), equation)
    return equation_func
    
def equationx2(equ):
    x1 = sp.Symbol('x1')
    x3 = sp.Symbol('x3')
    equation = sp.sympify(equ)
    equation_func = sp.lambdify((x1, x3), equation)
    return equation_func
    
def equationx1(equ):
    x2 = sp.Symbol('x2')
    x3 = sp.Symbol('x3')
    equation = sp.sympify(equ)
    equation_func = sp.lambdify((x2, x3), equation)
    return equation_func

def Differentiation(equ,var):
    x=sp.Symbol('x')
    equation=sp.sympify(equ)
    #function make diff to equation
    equation_deriv=sp.diff(equation,x)
    #subs var into equation after diff
    return equation_deriv.subs(x,var)

def SecondDifferentiation(equ,var):
    x=sp.Symbol('x')
    equation=sp.sympify(equ)
    #function make  second diff to equation
    equation_deriv=sp.diff(equation,x,2)
    return equation_deriv.subs(x,var)

def IsDoubleRoot(equ,var):
    flag=False
    function=equation(equ)
    diff=Differentiation(equ,var)
    if function(var)==diff:
        flag=True
    return flag 

def bisection_method():
    equ=input("Enter the equation in term of x: ")
    function=equation(equ)
    A=float(input("Enter the value a: "))
    B=float(input("Enter the value b: "))
    accuracy=float(input("Enter the value of your accuracy: "))
    iterations=(math.log(1/accuracy))/math.log(2)
    #print(iterations)
    #test condition if we should apply the method divergent
    if function(A)*function(B)>=0:
        print("Bisection method failed")
        return None
    print("Number of steps needed to achieve an accuracy of ",accuracy,"are ",math.ceil(iterations))
    iteration=0
    for i in range(math.ceil(iterations)):
        #midpoint
        C=(A+B)/2
        iteration+=1
        print("At iteartion [",iteration,"]: ",C)
        #positive replace by A
        if function(A)*function(C)>=0:
            A=C
        else:
            B=C
    print("The solution of the equation is ",C)
    return C


def NewtonRaphson_Method():
    equ=input("Enter the equation in term of x: ")
    function=equation(equ)
    X=float(input("Enter the initial value of x: "))
    #test for double root
    if IsDoubleRoot(equ,X):
        print("This function has a double root")
    else:
        #rule newton 
        equ_deriv=Differentiation(equ,X)
        Xn=X-(function(X)/equ_deriv)
        i=0
        flag=True
        while flag and i <1000 :
            Xn=float(X-(function(X)/equ_deriv))
            if Xn==X:
                flag=False
            else:
                print("At iteration [",i+1,"]: ",Xn)
                #change each x with new x
                X=Xn
                i+=1
        print("The solution is Convergence to ",Xn," at the ",i,"th iteration")



def Hally_Method():
    equ=input("Enter the equation in term of x: ")
    function=equation(equ)
    X=float(input("Enter the initial value of x: "))
    equ_deriv1=Differentiation(equ,X)
    equ_deriv2=SecondDifferentiation(equ,X)
    #rule of hally
    Xn=X-((function(X)*equ_deriv1))/((equ_deriv1**2)-(function(X)*equ_deriv2))
    i=0
    flag=True
    while flag and i <1000 :
        Xn=float(X-((function(X)*equ_deriv1))/((equ_deriv1**2)-(function(X)*equ_deriv2)))
        if Xn==X:
            flag=False
        else:
            print("At iteration [",i+1,"]: ",Xn)
            #update each x
            X=Xn
            i+=1
    print("The solution is Convergence to ",Xn," at the ",i,"th iteration")

def FixedPointIteration_Method():
    flag=True
    equ=[]
    i=0
    initial_points=[]
    while flag:
        equ.append(input("Enter the equation in term of x: "))
        choice=input("If you want to enter another equation write ""yes"" else press no ")
        if choice=="yes":
            i+=1
            continue
        if choice=="no":
            flag=False
    X=float(input("Enter the initial value of x: "))
    length=len(equ)
    if length==1 and abs(Differentiation(equ[i],X))>=1 :
        print("The solution is divergence ")
    elif length==1 and abs(Differentiation(equ[i],X))<1:
        print(abs(Differentiation(equ[i],X)))
        function=equation(equ[i])
        xn=function(X)
        j=0
        flag2=True
        while flag2 and j <1000 :
            xn=float(function(X))
            if xn==X:
                flag2=False
            else:
                print("At iteration [",j+1,"]: ",xn)
                X=xn
                j+=1
        print("The solution is Convergence to ",xn," at the ",j,"th iteration")
    elif length>1:
        convergence=abs(Differentiation(equ[0],X))
        function=equation(equ[0])
        function_String=equ[0]
        for j in range(length):
            if abs(Differentiation(equ[j],X))<convergence:
                convergence=abs(Differentiation(equ[j],X))
                function=equation(equ[j])
                function_String=equ[j]
        print("\nthe equation: ",function_String  ," will give the best solution")
        xn=function(X)
        j=0
        flag2=True
        while flag2 and j <1000 :
            xn=float(function(X))
            if xn==X:
                flag2=False
            else:
                print("At iteration [",j+1,"]: ",xn)
                X=xn
                j+=1
        print("The solution is Convergence to ",xn," at the ",j,"th iteration")


def Secant_Method():
    equ=input("Enter the equation in term of x: ")
    function=equation(equ)
    x0=float(input("Enter the value first X: "))
    x1=float(input("Enter the value second X: "))
    #rule secant
    Xn = x1 - function(x1) * (x1 - x0) / (function(x1) - function(x0))
    i=0
    flag=True
    while flag and i <1000 :
        Xn =float(x1 - function(x1) * (x1 - x0) / (function(x1) - function(x0)))
        if Xn==x1:
            flag=False
        else:
            print("At iteration [",i+1,"]: ",Xn)
            #switch xs
            x0=x1
            x1=Xn
            i+=1   
    print("The solution is Convergence to ",Xn," at the ",i,"th iteration")


def False_Position_Method():
    equ=input("Enter the equation in term of x: ")
    function=equation(equ)
    A=float(input("Enter the value A: "))
    B=float(input("Enter the value B: "))
    #test for divergent
    if function(A)*function(B)>=0:
        print("False position method failed")
        return None
    i=0
    flag=True
    while flag and i <1000 :
        C =float(B - (function(B) * (B -A)) / (function(B) - function(A)))
        i+=1
        #C start to be constant
        if(C==A or C==B):
            flag=False
        print("At iteration [",i,"]: ",C)
        # replace with variable of same sign of C
        if function(A)*function(C)>=0:
            A=C
        else:
            B=C
    print("The solution is Convergence to ",C," at the ",i,"th iteration")
    return C

def Jacobi_Method():
    equ1=input("Enter the equation1 in term of x1,x2,x3,=: ")
    equ2=input("Enter the equation2 in term of x1,x2,x3,=: ")
    equ3=input("Enter the equation3 in term of x1,x2,x3,=: ")
    # make three equation each one for x1,x2,x3 and then test for condition for jacobi
    function1=equationx1(isolatex1(equ1))
    function2=equationx2(isolatex2(equ2))
    function3=equationx3(isolatex3(equ3))
    boolean=test_coefficients(equ1,equ2,equ3)
    #condition true
    if(boolean==False):
        x1=float(input("Enter the value of initial x1: "))
        x2=float(input("Enter the value of initial x2: "))
        x3=float(input("Enter the value of initial x3: "))
        #replace each function with it initial variables
        x1new=function1(x2,x3)
        i=1
        #generate the new variables x1,x2,x3
        print("At iteration ",i," x1= ",x1new)
        x2new=function2(x1,x3)
        print("At iteration ",i," x2= ",x2new)
        x3new=function3(x1,x2)
        print("At iteration ",i," x3= ",x3new)
        flag=True
        while flag and i <1000 :
            x1=x1new
            x2=x2new
            x3=x3new
            i+=1
            #subs new variables in function
            x1new=function1(x2,x3)
            print("At iteration ",i," x1= ",x1new)
            x2new=function2(x1,x3)
            print("At iteration ",i," x2= ",x2new)
            x3new=function3(x1,x2)
            print("At iteration ",i," x3= ",x3new)
            #test when variables start to convergent
            if check_decimal_after_dot(x1new) and check_decimal_after_dot(x2new) and check_decimal_after_dot(x3new):
                flag=False
        print("The solution is Convergence to x1= ",x1new," x2= ",x2new," x3= ",x3new," at the ",i,"th iteration") 
    else:
        print("Jacobi method failed")
        

def GaussSeidel_Method():
    equ1=input("Enter the equation1 in term of x1,x2,x3,=: ")
    equ2=input("Enter the equation2 in term of x1,x2,x3,=: ")
    equ3=input("Enter the equation3 in term of x1,x2,x3,=: ")
    function1=equationx1(isolatex1(equ1))
    function2=equationx2(isolatex2(equ2))
    function3=equationx3(isolatex3(equ3))
    x1=float(input("Enter the value of initial x1: "))
    x2=float(input("Enter the value of initial x2: "))
    x3=float(input("Enter the value of initial x3: "))
    i=0
    flag=True
    while flag and i <1000 :
        i+=1
        #update each variable instantly with new one in same iteration
        x1=function1(x2,x3)
        print("At iteration ",i," x1= ",x1)
        x2=function2(x1,x3)
        print("At iteration ",i," x2= ",x2)
        x3=function3(x1,x2)
        print("At iteration ",i," x3= ",x3)
        #close to convergent
        if check_decimal_after_dot(x1) and check_decimal_after_dot(x2) and check_decimal_after_dot(x3):
            flag=False
    print("The solution is Convergence to x1= ",x1," x2= ",x2," x3= ",x3," at the ",i,"th iteration") 
     
def regression():
    n = int(input("Enter the number of data points: "))
    x = []
    y = []
    sumX = 0
    sumY = 0
    sumXsq = 0
    sumXY = 0
    for i in range(n):
        x_val = float(input(f"Enter the value of x[{i}]: "))
        y_val = float(input(f"Enter the value of y[{i}]: "))
        x.append(x_val)
        y.append(y_val)
        sumXsq += x_val * x_val
        sumXY += x_val * y_val
   
    sumX = sum(x)
    sumY = sum(y)
    print("x list:", x)
    print("y list:", y)
    print("sum X:", sumX)
    print("sum y:", sumY)
    print("sum x^2:", sumXsq)
    print("sum X*Y:", sumXY)
    print("Equation1= ",n,"a+",sumX,"b=",sumY)
    print("Equation2= ",sumX,"a+",sumXsq,"b=",sumXY)
    meanX = sumX / n
    meanY = sumY / n
    #method of least squares
    b = (sumXY - n * meanX * meanY) / (sumXsq - n * meanX ** 2)
    a = meanY - b * meanX
    print("Coefficient a:", a)
    print("Coefficient b:", b)
    print("y=",a,"+(",b,")x")

  
#FixedPointIteration_Method()
#newton_raphson_gui()
#NewtonRaphson_Method()
#bisection_method()
#GaussSeidel_Method()        
#Jacobi_Method()    
#False_Position_Method()
#Secant_Method()
#Hally_Method()
#regression()

