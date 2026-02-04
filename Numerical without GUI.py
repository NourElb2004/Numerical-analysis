import sympy as sp
import numpy as np
import math
from sympy import symbols, Eq, solve, sympify,expand,diff,integrate


def equation(equ):
    #equation in terms of symbol
    x = sp.Symbol('x')
    #convert string into sympy expression
    equation = sp.sympify(equ)
    #callable function to evaluate symbolic expression numerically
    equation_func = sp.lambdify(x, equation)
    return equation_func
def equation_x_y(equ):
    #equation in terms of symbol
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    #convert string into sympy expression
    equation = sp.sympify(equ)
    #callable function to evaluate symbolic expression numerically
    equation_func = sp.lambdify((x,y),equation)
    return equation_func
def check_decimal_after_dot(number):
    return str(number).endswith('.0')
def FourthDifferentiation(equ,var):
    x = sp.Symbol('x')
    equation = sp.sympify(equ)
    fourth_derivative = sp.diff(equation, x, 4)
    return fourth_derivative.subs(x,var)
def Integration(equ, a, b):
    x = sp.Symbol('x')
    equation = sp.sympify(equ)
    # Integrate the equation with respect to x
    integral = sp.integrate(equation, (x, a, b))
    return integral
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



def Aitkens_process_Newton(equ,X,Xn,i):
    if X==Xn:
        print("The solution is Convergence to ",Xn," at the ",i,"th iteration")
        return None
    if X==0 or Xn==0:
        print("The solution is Convergence to ",Xn," at the ",i,"th iteration")
        return None
    i=i+1
    initial_points=[]
    function=equation(equ)
    equ_deriv=Differentiation(equ,X)
    if(equ_deriv==0):
        print("NewtonRaphson method failed")
        return None
    Xn=X-(function(X)/equ_deriv)
    for i in range(3): 
        Xn=float(X-(function(X)/equ_deriv))
        initial_points.append(Xn)
        X=Xn
    x3=initial_points[2]
    deltaX1=initial_points[1]-initial_points[0]
    deltax2=initial_points[2]-initial_points[1]
    deltasquarex1=deltax2-deltaX1
    if deltasquarex1==0:
        print("The solution is Convergence to ",Xn," at the ",i,"th iteration")
        return None
    x0new=x3-(math.pow(deltax2,2)/deltasquarex1)
    return Aitkens_process_Newton(equ,x0new,Xn,i)
    

def NewtonRaphson_Method():
    initial_point=[]
    equ=input("Enter the equation in term of x: ")
    function=equation(equ)
    X=float(input("Enter the initial value of x: "))
    x_initial=X
    doubleroot=False
    #test for double root
    if IsDoubleRoot(equ,X):
        print("This function has a double root")
        X=0.01
        doubleroot=True
        choice=input("If you want to continue with newton raphson write continue \nIf you want to use Aitken's process write Aitken ")
        if choice=="Aitken":
            equ_deriv=Differentiation(equ,X)
            Xn=X-(function(X)/equ_deriv)
            Aitkens_process_Newton(equ,X,Xn,i=0)
            return None
        elif choice=="continue":
           pass
    #Newton rule
    equ_deriv=Differentiation(equ,X)
    if(equ_deriv==0):
        print("NewtonRaphson method failed")
        return None
    Xn=X-(function(X)/equ_deriv)
    i=0
    flag=True
    while flag and i <1000 :
        Xn=float(X-(function(X)/equ_deriv))
        if Xn==X:
            flag=False
        else:
            initial_point.append(Xn)
            print("At iteration [",i+1,"]: ",Xn)
            #change each x with new x
            X=Xn
            i+=1
    print("The solution is Convergence to ",Xn," at the ",i,"th iteration")
    choice2=input("If you want to Try Aitken's process write yes ")
    if choice2=="yes":
        if(doubleroot==True):
              X=0.01
              equ_deriv=Differentiation(equ,X)
              Xn=X-(function(X)/equ_deriv)
              Aitkens_process_Newton(equ,X,Xn,i=0)
        else:
            Aitkens_process_Newton(equ,x_initial,initial_point[0],i=0)
def Aitkens_Process_Fixed(equ,X,Xn,i):
    if X==Xn:
        print("The solution is Convergence to ",Xn," at the ",i,"th iteration")
        return None
    if X==0 or Xn==0:
        print("The solution is Convergence to ",Xn," at the ",i,"th iteration")
        return None
    i=i+1
    initial_points=[]
    Xn=equ(X)
    for i in range(3):
        Xn=float(equ(X))
        initial_points.append(Xn)
        X=Xn
    x3=initial_points[2]
    deltaX1=initial_points[1]-initial_points[0]
    deltax2=initial_points[2]-initial_points[1]
    deltasquarex1=deltax2-deltaX1
    if deltasquarex1==0:
        print("The solution is Convergence to ",Xn," at the ",i,"th iteration")
        return None
    x0new=x3-(math.pow(deltax2,2)/deltasquarex1)
    return Aitkens_Process_Fixed(equ,x0new,Xn,i)


            

def Hally_Method():
    equ=input("Enter the equation in term of x: ")
    function=equation(equ)
    X=float(input("Enter the initial value of x: "))
    equ_deriv1=Differentiation(equ,X)
    equ_deriv2=SecondDifferentiation(equ,X)
    if(equ_deriv1==0 and equ_deriv2==0):
        print("Hally method failed")
        return None
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
    initial_point=[]
    equ=[]
    i=0
    n=int(input("Enter number of equations: "))
    for i in range(n):
        equ.append(input("Enter the equation in term of x: "))
    X=float(input("Enter the initial value of x: "))
    x_initial=X
    if n==1 and abs(Differentiation(equ[i],X))>=1 :
        print("The solution is divergence ")
    elif n==1 and abs(Differentiation(equ[i],X))<1:
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
                initial_point.append(xn)
                print("At iteration [",j+1,"]: ",xn)
                X=xn
                j+=1
            Xn=initial_point[0]
        print("The solution is Convergence to ",xn," at the ",j,"th iteration")
        choice2=input("If you want to Try Aitken's process write yes ")
        if choice2=="yes":
            Aitkens_Process_Fixed(function,x_initial,Xn,i=0)
    elif n>1:
        convergence=abs(Differentiation(equ[0],X))
        function=equation(equ[0])
        function_String=equ[0]
        for j in range(n):
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
                initial_point.append(xn)
                print("At iteration [",j+1,"]: ",xn)
                X=xn
                j+=1
        Xn=initial_point[0]
        print("The solution is Convergence to ",xn," at the ",j,"th iteration")
        choice2=input("If you want to Try Aitken's process write yes ")
        if choice2=="yes":
            Aitkens_Process_Fixed(function,x_initial,Xn,i=0)

def Secant_Method():
    equ=input("Enter the equation in term of x: ")
    function=equation(equ)
    x0=float(input("Enter the value first X: "))
    x1=float(input("Enter the value second X: "))
    if function(x0)*function(x1)>=0:
        print("Secant method failed")
        return None
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

def equationx(equ):
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    equation = sp.sympify(equ)
    return equation
def equationmu(equ):
    mu1, mu2, mu3 = sp.symbols('mu1 mu2 mu3')
    equation = sp.sympify(equ)
    return equation

def Jacobi_Method_Non_Linear_Equations():
    equ1=input("Enter the equation1 in term of x1,x2,x3: ")
    equ2=input("Enter the equation2 in term of x1,x2,x3: ")
    equ3=input("Enter the equation3 in term of x1,x2,x3: ")
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    x11=float(input("Enter the value of initial x1: "))
    x22=float(input("Enter the value of initial x2: "))
    x33=float(input("Enter the value of initial x3: "))
    numbers = [x11,x22,x33]
    matrixXs = sp.Matrix([numbers])
    print(matrixXs)
    function1=equationx(equ1)
    function2=equationx(equ2)
    function3=equationx(equ3)
    List_diff1=[sp.diff(function1,x1),sp.diff(function1,x2),sp.diff(function1,x3)]
    List_diff2=[sp.diff(function2,x1),sp.diff(function2,x2),sp.diff(function2,x3)]
    List_diff3=[sp.diff(function3,x1),sp.diff(function3,x2),sp.diff(function3,x3)]
    #print(List_diff)
    matrix_diff = sp.Matrix([List_diff1,List_diff2,List_diff3])
    print('Matrix After Differentiation: ',matrix_diff)
    rows, cols = matrix_diff.shape
    for i in range(1):
        List_sub=[(function1.subs({x1: x11, x2:x22, x3: x33}))*-1,(function2.subs({x1: x11, x2: x22, x3: x33}))*-1,(function3.subs({x1: x11, x2: x22, x3: x33}))*-1]
        matrix_sub=sp.Matrix([List_sub])
        print('Matrix Substitution in Equations Before Differentiation: ',matrix_sub)
        List_diff_sub=[]
        for i in range(rows):
            for j in range(cols):
                f=equationx(matrix_diff[i, j])
                x_sub=(f.subs({x1: x11, x2: x22, x3: x33})).evalf()
                List_diff_sub.append(x_sub)
        List_sub1=List_diff_sub[0:3]
        List_sub2=List_diff_sub[3:6]
        List_sub3=List_diff_sub[6:9]
        #print(List_diff_sub)
        matrix_diff_sub = sp.Matrix([List_sub1,List_sub2,List_sub3])
        print('Matrix Substitution After Differentiation: ',matrix_diff_sub)     
        mu=sp.Matrix([["mu1"],["mu2"],["mu3"]])
        matrix=matrix_diff_sub*mu
        print(matrix_sub)
        Combined_matrix=matrix.T-matrix_sub
        print('Matrix Mus Equations: ',Combined_matrix)
        mu1, mu2, mu3 = sp.symbols('mu1 mu2 mu3')
        solutions_list=[]
        equations_list = Combined_matrix.tolist()
        solutions = []
        my_array=np.array(equations_list)
        for equation in equations_list:
            solution = sp.solve(equation, (mu1, mu2, mu3))
            for key, value in solution.items():
                solutions.append(value)  
        #print(solutions)
        matrix_mus=sp.Matrix([solutions])
        x_new=matrixXs+matrix_mus
        print('Matrix of New X: ',x_new)
        x11=solutions[0]
        x22=solutions[1]
        x33=solutions[2]
        x_new=matrixXs


def Newton_forward():
    size = int(input("Enter the number of values: "))
    x = np.array([])
    y = np.array([])
    x_interp = np.array([])
    fail=False
    # Input x and y values in the table
    for i in range(size):
        xn = float(input("Enter value of x{}: ".format(i+1)))
        x = np.append(x, xn)
        yn = float(input("Enter value of y{}: ".format(i+1)))
        y = np.append(y, yn)
    # Print Values of x, y 
    print("x:", x)
    print("y:", y)
    interval = x[1] - x[0]
    for i in range(1, len(x) - 1):
        if x[i + 1] - x[i] != interval:
            fail=True
    if(fail==True):
         print("Newton Forward method failed as interval between x values is not constant")
         return None
    else:
        size_x_interp = int(input("Enter the number of values of x-coordinates where interpolation is desired: "))
        for i in range(size_x_interp):
            x_interp_value = float(input("Enter value of x{}: ".format(i+1)))
            x_interp = np.append(x_interp, x_interp_value)
        # Print x values for interpolation
        print("x_interp:", x_interp)
        n = len(x)
        h = x[1] - x[0]
        # Initialize forward differences matrix by zeros
        forward_diff = np.zeros((n, n))
        # Set the first column of forward differences matrix to y-values
        forward_diff[:, 0] = y
        for j in range(1, n):
            for i in range(n - j):
                forward_diff[i, j] = forward_diff[i+1, j-1] - forward_diff[i, j-1]
        p = symbols('p')
        polynomial_eq = forward_diff[0, 0]
        for j in range(1, n):
            term = 1
            for k in range(j):
                term *= (p - k)
                term /= (k + 1)
            polynomial_eq += term * forward_diff[0, j]
        # Print polynomial equation
        print("Polynomial equation: ", polynomial_eq)
        print("Simplified Polynomial equation before substitution:", expand(polynomial_eq))
        first_derivative = diff(polynomial_eq, p)
        second_derivative = diff(polynomial_eq, p,2)
        for i in range(size_x_interp):
            print("Polynomial equation at point ",x_interp[i],' is ', polynomial_eq.subs(p,(x_interp[i]-x[0])/h))
            print("First derivative at point ",x_interp[i],' is ',(1/h)*(first_derivative.subs(p,(x_interp[i]-x[0])/h)))
            print("Second derivative at point ",x_interp[i],' is ',(1/math.pow(h,2))*(second_derivative.subs(p,(x_interp[i]-x[0])/h)))
        a = float(input("Enter the lower limit of integration: "))
        b = float(input("Enter the upper limit of integration: "))
        integral = integrate(polynomial_eq, (p, (a-x[0])/h,(b-x[0])/h))
        print("Integration :",h*integral)

def Newton_backward():
    size = int(input("Enter the number of values: "))
    x = np.array([])
    y = np.array([])
    x_interp = np.array([])
    fail=False
    # Input x and y values in the table
    for i in range(size):
        xn = float(input("Enter value of x{}: ".format(i+1)))
        x = np.append(x, xn)
        yn = float(input("Enter value of y{}: ".format(i+1)))
        y = np.append(y, yn)
    # Print Values of x, y 
    print("x:", x)
    print("y:", y)
    interval = x[1] - x[0]
    for i in range(1, len(x) - 1):
        if x[i + 1] - x[i] != interval:
            fail=True
    if(fail==True):
        print("Newton Backward method failed as interval between x values is not constant")
        return None
    else:
        size_x_interp = int(input("Enter the number of values of x-coordinates where interpolation is desired: "))
        for i in range(size_x_interp):
            x_interp_value = float(input("Enter value of x{}: ".format(i+1)))
            x_interp = np.append(x_interp, x_interp_value)
        # Print x values for interpolation
        print("x_interp:", x_interp)
        n = len(x)
        h = x[1] - x[0]
        # Initialize backward differences matrix by zeros
        backward_diff = np.zeros((n, n))
        # Set the last column of backward differences matrix to y-values
        backward_diff[:, 0] = y
        for j in range(1, n):
            for i in range(j,n):
                backward_diff[i,j] = backward_diff[i,j-1] - backward_diff[i-1,j - 1]
        s = symbols('s')
        polynomial_eq = backward_diff[-1, 0]
        for j in range(1, n):
            term = 1
            for k in range(j):
                term *= (s + k)
                term /= (k + 1)
            polynomial_eq += term * backward_diff[-1,j]
        # Print polynomial equation
        print("Polynomial equation: ", polynomial_eq)
        print("Simplified Polynomial equation before substitution:", expand(polynomial_eq))
        first_derivative = diff(polynomial_eq, s)
        second_derivative = diff(polynomial_eq, s,2)
        for i in range(size_x_interp):
            print("Polynomial equation at point ",x_interp[i],' is ',polynomial_eq.subs(s,(x_interp[i]-x[-1])/h))
            print("First derivative at point ",x_interp[i],' is ',(1/h)*(first_derivative.subs(s,(x_interp[i]-x[-1])/h)))
            print("Second derivative at point ",x_interp[i],' is ',(1/math.pow(h,2))*(second_derivative.subs(s,(x_interp[i]-x[-1])/h)))
        a = float(input("Enter the lower limit of integration: "))
        b = float(input("Enter the upper limit of integration: "))
        integral = integrate(polynomial_eq, (s, (a-x[0])/h,(b-x[0])/h))
        print("Integration :",h*integral)
   
def Newton_divided_difference_interpolation():
    size=int(input("Enter the number of values: "))
    x=np.array([])
    y=np.array([])
    x_interp=np.array([])
    # Input x and y values in the table
    for i in range(size):
        xn=float(input("Enter value of x{}: ".format(i+1)))
        x=np.append(x,xn)
        yn=float(input("Enter value of y{}: ".format(i+1)))
        y=np.append(y,yn)
    # Print Values of x, y 
    print(x)
    print(y)
    size_x_interp=int(input("Enter the number of values of x-coordinates where interpolation is desired: "))
    for i in range(size_x_interp):
        x_interp_value=float(input("Enter value of x{}: ".format(i+1)))
        x_interp=np.append(x_interp,x_interp_value)
    # Print x values interpolation
    print(x_interp)
    n = len(x)
    t = symbols('t')
    # Initialize Function matrix by zeros
    F = [[0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        F[j][0] = y[j]
    for j in range(1, n):
        for i in range(n - j):
            F[i][j] = (F[i+1][j-1] - F[i][j-1]) / (x[i+j] - x[i])
    # Calculate the polynomial equation
    polynomial_eq = F[0][0]
    for j in range(1, n):
        term = 1
        for k in range(j):
            term *= (t - x[k])
        polynomial_eq += F[0][j] * term
    # Print polynomial equation
    print("Polynomial equation before substitution:", polynomial_eq)
    print("Simplified Polynomial equation before substitution:", expand(polynomial_eq))
    first_derivative = diff(polynomial_eq, t)
    second_derivative = diff(polynomial_eq, t,2)
    for i in range(size_x_interp):
        print("Polynomial equation at point ",x_interp[i],' is ',polynomial_eq.subs(t, x_interp[i]))
        print("First derivative at point ",x_interp[i],' is ',first_derivative.subs(t,(x_interp[i])))
        print("Second derivative at point ",x_interp[i],' is ',second_derivative.subs(t,(x_interp[i])))
    a = float(input("Enter the lower limit of integration: "))
    b = float(input("Enter the upper limit of integration: "))
    integral = integrate(polynomial_eq, (t, a,b))
    print("Integration :",integral)
    
def LaGrange():
    size = int(input("Enter the number of values: "))
    x = np.array([])
    y = np.array([])
    x_interp = np.array([])
    # Input x and y values in the table
    for i in range(size):
        xn = float(input("Enter value of x{}: ".format(i+1)))
        x = np.append(x, xn)
        yn = float(input("Enter value of y{}: ".format(i+1)))
        y = np.append(y, yn)
    # Print Values of x, y 
    print("x:", x)
    print("y:", y)
    size_x_interp = int(input("Enter the number of values of x-coordinates where interpolation is desired: "))
    for i in range(size_x_interp):
        x_interp_value = float(input("Enter value of x{}: ".format(i+1)))
        x_interp = np.append(x_interp, x_interp_value)
    # Print x values for interpolation
    print("x_interp:", x_interp)
    t = symbols('t')
    n = len(x)
    m = len(x_interp)
    interpolated_y = np.zeros(m)
    polynomial_eq = 0
    for i in range(n):
        term = 1
        for k in range(n):
            if k != i:
                term *= (t - x[k]) / (x[i] - x[k])
        polynomial_eq += y[i] * term
    # Print polynomial equation 
    print("Polynomial equation: ", polynomial_eq)
    print("Simplified Polynomial equation before substitution:", expand(polynomial_eq))
    first_derivative = diff(polynomial_eq, t)
    second_derivative = diff(polynomial_eq, t,2)
    for i in range(size_x_interp):
        print("Polynomial equation at point ",x_interp[i],' is ', polynomial_eq.subs(t,x_interp[i]))
        print("First derivative at point ",x_interp[i],' is ',first_derivative.subs(t,x_interp[i]))
        print("Second derivative at point ",x_interp[i],' is ',second_derivative.subs(t,x_interp[i]))
    a = float(input("Enter the lower limit of integration: "))
    b = float(input("Enter the upper limit of integration: "))
    integral = integrate(polynomial_eq, (t, (a,b)))
    print("Integration :",integral)

def Trapezoidal():
    # n: number of values in table -1
    n=int(input("Enter the n: "))
    equ=input("Enter the equation in term of x: ")
    function=equation(equ)
    print("Enter boundaries")
    a=float(input("Enter the value of lower limit: "))
    b=float(input("Enter the value of upper limit: "))
    h=(b-a)/n
    print ("h=",h)
    x = np.linspace(a, b, n+1)
    y = function(x)
    print("X:",x)
    print("Y:",y)
    integral_T = h * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]))
    print("Trapezoidal Integral= ",integral_T)
    exact_Integral=Integration(equ,a,b)
    print("Exact Integral= ",exact_Integral)
    absolute_error=abs(exact_Integral-integral_T)
    relative_error=abs(absolute_error/exact_Integral)
    print("Absolute Error: ",'{:.20f}'.format(absolute_error))
    print("Relative Error: ",'{:.20f}'.format(relative_error))
    sec_Differentiation=SecondDifferentiation(equ,a)
    tr_T= ((h**2*(b-a))/12)*sec_Differentiation
    print("Tranction Error of Trapezoidal : ",'{:.20f}'.format(abs(tr_T)))
    if(sec_Differentiation<0):
         print(integral_T," <=  Exact integration <= ",(integral_T+abs(tr_T)))
    else:
         print((integral_T-abs(tr_T))," <=  Exact integration <= ",integral_T)
        
    
def Simpson():
    # n: number of values in table -1
    n=int(input("Enter the n: "))
    equ=input("Enter the equation in term of x: ")
    function=equation(equ)
    print("Enter boundaries")
    a=float(input("Enter the value of lower limit: "))
    b=float(input("Enter the value of upper limit: "))
    h=(b-a)/n
    print ("h=",h)
    x = np.linspace(a, b, n+1)
    y = function(x)
    print("X:",x)
    print("Y:",y)
    integral_S = h / 3 * (y[0] + y[-1] + 2 * np.sum(y[2:-2:2]) + 4 * np.sum(y[1:-1:2]))
    print("Simpson's Integral= ", integral_S)
    exact_Integral=Integration(equ,a,b)
    print("Exact Integral= ",exact_Integral)
    absolute_error=abs(exact_Integral-integral_S)
    relative_error=abs(absolute_error/exact_Integral)
    print("Absolute Error: ",'{:.20f}'.format(absolute_error))
    print("Relative Error: ",'{:.20f}'.format(relative_error))
    fourth_differentiation=FourthDifferentiation(equ,a)
    tr_S= ((h**4*(b-a))/180)*fourth_differentiation
    print("Tranction Error of Simpson : ",'{:.20f}'.format(abs(tr_S)))
    if(fourth_differentiation<0):
         print(integral_S," <=  Exact integration <= ",(integral_S+abs(tr_S)))
    else:
         print((integral_S-abs(tr_S))," <=  Exact integration <= ",integral_S)
    
def forward_difference1(x, y, h):
    derivatives = []
    n = len(x)
    for i in range(n - 1):
        derivative = (y[i+1] - y[i]) / h
        derivatives.append((x[i], "Forward", derivative))
    return derivatives

def backward_difference1(x, y, h):
    derivatives = []
    n = len(x)
    for i in range(n - 1, 0, -1):
        derivative = (y[i] - y[i-1]) / h
        derivatives.append((x[i], "Backward", derivative))
    return derivatives

def Two_points_Formula():
    size = int(input("Enter the number of values: "))
    x = np.array([])
    y = np.array([])
    x_interp = np.array([])
    # Input x and y values in the table
    for i in range(size):
        xn = float(input("Enter value of x{}: ".format(i+1)))
        x = np.append(x, xn)
        yn = float(input("Enter value of y{}: ".format(i+1)))
        y = np.append(y, yn)
    # Print Values of x, y 
    print("x:", x)
    print("y:", y)
    h= float(input("Enter the value of h: "))
    forward_diff = forward_difference1(x, y, h)
    backward_diff = backward_difference1(x, y, h)  
    print("Forward Differences: ")
    for point in forward_diff:
        print(point)
    print("\nBackward Differences: ")
    for point in backward_diff:
        print(point)

def forward_difference(x, y, h):
    derivatives = []
    n = len(x)
    for i in range(n - 2):
        derivative = (3*y[i]- 4*y[i+1] +y[i+2]) / (-1*h*2)
        derivatives.append((x[i], "Forward", derivative))
    return derivatives

def backward_difference(x, y, h):
    derivatives = []
    n = len(x)
    for i in range(n-1 , 1, -1):
        derivative = (3*y[i]- 4*y[i-1] +y[i-2]) / (h*2)
        derivatives.append((x[i], "Backward", derivative))
    return derivatives

def central_difference(x, y, h):
    derivatives = []
    n = len(x)
    for i in range(1, n - 1):
        derivative = (y[i+1] -y[i-1]) / (h*2)
        derivatives.append((x[i], "Central", derivative))
    return derivatives

def Three_points_Formula():
    size = int(input("Enter the number of values: "))
    x = np.array([])
    y = np.array([])
    x_interp = np.array([])
    # Input x and y values in the table
    for i in range(size):
        xn = float(input("Enter value of x{}: ".format(i+1)))
        x = np.append(x, xn)
        yn = float(input("Enter value of y{}: ".format(i+1)))
        y = np.append(y, yn)
    # Print Values of x, y 
    print("x:", x)
    print("y:", y)
    h= float(input("Enter the value of h: "))
    forward_diff = forward_difference(x, y, h)
    backward_diff = backward_difference(x, y, h)
    central_diff = central_difference(x, y, h)
    print("Forward Differences:")
    for point in forward_diff:
        print(point)
    print("\nBackward Differences:")
    for point in backward_diff:
        print(point)
    print("\nCentral Differences:")
    for point in central_diff:
        print(point)


def Runge_kutta_3():
    equ=input("Enter the differential equation in term of x and y: ")
    function=equation_x_y(equ)
    x0=float(input("Enter the value of initial x: "))
    y0=float(input("Enter the value of initial y: "))
    print("Enter boundaries")
    a=float(input("Enter the value of lower limit: "))
    b=float(input("Enter the value of upper limit: "))
    n=int(input("Enter number of iterations (n): "))
    h=(b-a)/n
    print ("n=",n)
    print ("h=",h)
    x_values = [x0]
    y_values = [y0]

    for i in range(int(n)):
        k1 = h * function(x_values[-1], y_values[-1])
        k2 = h * function(x_values[-1] + h/2, y_values[-1] + k1/2)
        k3 = h * function(x_values[-1] + h/2, y_values[-1] + k2/2)
        k4 = h * function(x_values[-1] + h, y_values[-1] + k3)
        y_next = y_values[-1] + (k1 + 2*k2 + 2*k3+ k4) / 6
        y_values.append(y_next)
        x_next = round(x_values[-1] + h, 10) 
        x_values.append(x_next)
    for x, y in zip(x_values, y_values):
        print(f"x = {x}, y = {y}")
        
def Euler():
    equ = input("Enter the differential equation in term of x and y: ")
    function = equation_x_y(equ)
    x0=float(input("Enter the value of initial x: "))
    y0=float(input("Enter the value of initial y: "))
    print("Enter boundaries")
    a=float(input("Enter the value of lower limit: "))
    b=float(input("Enter the value of upper limit: "))
    n=int(input("Enter number of iterations (n): "))
    h=(b-a)/n
    print ("n=",n)
    print ("h=",h)
    x_values = [x0]
    y_values = [y0]
    for i in range(n):
        y_next = y_values[-1] + h * function(x_values[-1], y_values[-1])
        y_values.append(y_next)
        x_next = round(x_values[-1] + h, 10) 
        x_values.append(x_next)
    for x, y in zip(x_values, y_values):
        print(f"x = {x}, y = {y}")
        


def Euler_Modified():
    equ = input("Enter the differential equation in term of x and y: ")
    function = equation_x_y(equ)
    x0=float(input("Enter the value of initial x: "))
    y0=float(input("Enter the value of initial y: "))
    print("Enter boundaries")
    a=float(input("Enter the value of lower limit: "))
    b=float(input("Enter the value of upper limit: "))
    n=int(input("Enter number of iterations (n): "))
    h=(b-a)/n
    print ("n=",n)
    print ("h=",h)
    x_values = [x0]
    y_values = [y0]
    for i in range(n):
        y_next = y_values[-1] + h * function(x_values[-1]+(h/2), y_values[-1]+(h/2)*function(x_values[-1], y_values[-1]))
        y_values.append(y_next)
        x_next = round(x_values[-1] + h, 10) 
        x_values.append(x_next)
    for x, y in zip(x_values, y_values):
        print(f"x = {x}, y = {y}")


#FixedPointIteration_Method()
#NewtonRaphson_Method()
#bisection_method()
#False_Position_Method()
#Secant_Method()
#Hally_Method()

#GaussSeidel_Method()        
#Jacobi_Method() 
#regression()
#Jacobi_Method_Non_Linear_Equations()


#LaGrange()
#Newton_forward()
#Newton_backward() 
#Newton_divided_difference_interpolation()

#Trapezoidal()
#Simpson() 

#Two_points_Formula()
#Three_points_Formula()

#Runge_kutta_3()
#Euler()
#Euler_Modified()
