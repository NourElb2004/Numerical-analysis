import sympy as sp
import math
import numpy as np
from sympy import symbols, Eq, solve, sympify,expand,diff,integrate
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from ttkthemes import ThemedTk
import tkinter as tk
from tkinter import ttk
def equation(equ):
    x = sp.Symbol('x')
    equation = sp.sympify(equ)
    equation_func = sp.lambdify(x, equation)
    return equation_func
def equation_x_y(equ):
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    equation = sp.sympify(equ)
    equation_func = sp.lambdify((x,y),equation)
    return equation_func
def check_decimal_after_dot(number):
    return str(number).endswith('.0')
def FourthDifferentiation(equ,var):
    x = sp.Symbol('x')
    equation = sp.sympify(equ)
    fourth_derivative = sp.diff(equation, x, 4)
    return fourth_derivative.subs(x,var)

def test_coefficients(equ1,equ2,equ3):
    flag=True
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    a= equ1.split('=')[0]
    b= equ2.split('=')[0]
    c= equ3.split('=')[0]
    equation1 = sp.sympify(a)
    equation2 = sp.sympify(b)
    equation3 = sp.sympify(c)
    equ1coeffs_x1 = equation1.coeff(x1)
    equ1coeffs_x2 = equation1.coeff(x2)
    equ1coeffs_x3 = equation1.coeff(x3)
    equ2coeffs_x1 = equation2.coeff(x1)
    equ2coeffs_x2 = equation2.coeff(x2)
    equ2coeffs_x3 = equation2.coeff(x3)
    equ3coeffs_x1 = equation3.coeff(x1)
    equ3coeffs_x2 = equation3.coeff(x2)
    equ3coeffs_x3 = equation3.coeff(x3)
    if(equ1coeffs_x1>(equ1coeffs_x2+equ1coeffs_x3) and equ2coeffs_x2>(equ2coeffs_x1+equ2coeffs_x3) and equ3coeffs_x3>(equ3coeffs_x2+equ3coeffs_x1)):
        flag=False
    return flag
def Integration(equ, a, b):
    x = sp.Symbol('x')
    equation = sp.sympify(equ)
    # Integrate the equation with respect to x
    integral = sp.integrate(equation, (x, a, b))
    return integral

def isolatex1(equ):
    x1, x2, x3 = symbols('x1 x2 x3')
    left_side, right_side = equ.split('=')
    equation = Eq(sympify(left_side) - sympify(right_side), 0)
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
    equation_deriv=sp.diff(equation,x)
    return equation_deriv.subs(x,var)

def SecondDifferentiation(equ,var):
    x=sp.Symbol('x')
    equation=sp.sympify(equ)
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
    def calculate():
        equ = equation_entry.get()
        A = float(A_entry.get())
        B = float(B_entry.get())
        accuracy = float(accuracy_entry.get())
        function = equation(equ)
        iterations = (math.log(1/accuracy))/math.log(2)
        if function(A)*function(B)>=0:
            messagebox.showerror("Error", "Bisection method failed")
            return None
        print(f"Number of steps needed to achieve an accuracy of {accuracy} are {math.ceil(iterations)}")
        iteration = 0
        for i in range(math.ceil(iterations)):
            C = (A+B)/2
            iteration+=1
            print(f"At iteartion [{iteration}]: {C}")
            if function(A)*function(C)>=0:
                A=C
            else:
                B=C
        solution_label.config(text=f"The solution is Convergence to {C} at the {iteration}th iteration")
    root = ThemedTk(theme="equilux")
    root.title("Bisection Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equation_label = ttk.Label(main_frame, text="Enter the equation in term of x:")
    equation_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry = ttk.Entry(main_frame, width=20)
    equation_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_label = ttk.Label(main_frame, text="Enter the value of A:")
    A_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the value of B:")
    B_label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    accuracy_label = ttk.Label(main_frame, text="Enter the accuracy:")
    accuracy_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    accuracy_entry = ttk.Entry(main_frame, width=20)
    accuracy_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=1, row=8, padx=10, pady=10, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    root.mainloop()
def Aitkens_process_Newton(equ,X,Xn,i):
    if X==Xn:
        return f"The solution is Convergence to {Xn} at the {i}th iteration"
    if X==0 or Xn==0:
        return f"The solution is Convergence to {Xn} at the {i}th iteration"
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
        return f"The solution is Convergence to {Xn} at the {i}th iteration"
    x0new=x3-(math.pow(deltax2,2)/deltasquarex1)
    return Aitkens_process_Newton(equ,x0new,Xn,i)

def Aitkens_Process_Fixed(equ,X,Xn,i):
    if X==Xn:
        return f"The solution is Convergence to {Xn} at the {i}th iteration"
    if X==0 or Xn==0:
        return f"The solution is Convergence to {Xn} at the {i}th iteration"
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
        return f"The solution is Convergence to {Xn} at the {i}th iteration"
    x0new=x3-(math.pow(deltax2,2)/deltasquarex1)
    return Aitkens_Process_Fixed(equ,x0new,Xn,i)

def NewtonRaphson_Method():
    def calculate():
        initial_point = []
        equ = equation_entry.get()
        function = equation(equ)
        X = float(initial_value_entry.get())
        x_initial = X
        doubleroot = False

        if IsDoubleRoot(equ, X):
            messagebox.showwarning("Warning", "This function has a double root")
            X = 0.01
            doubleroot = True
            choice = choice_entry.get()
            if choice == "Aitken":
                equ_deriv = Differentiation(equ, X)
                Xn = X - (function(X) / equ_deriv)
                Aitken = str(Aitkens_process_Newton(equ, X, Xn, i=0))
                solution_label.config(text=Aitken)
                return None
            elif choice == "continue":
                pass

        equ_deriv = Differentiation(equ, X)
        if equ_deriv == 0:
            messagebox.showerror("Error", "Newton-Raphson method failed")
            return None

        Xn = X - (function(X) / equ_deriv)
        i = 0
        flag = True
        while flag and i < 1000:
            Xn = float(X - (function(X) / equ_deriv))
            if Xn == X:
                flag = False
            else:
                initial_point.append(Xn)
                print(f"At iteration [{i+1}]: {Xn}")
                X = Xn
                i += 1
        solution_label.config(text=f"The solution is convergence to {Xn} at the {i}th iteration")

        choice2 = choice2_entry.get()
        if choice2 == "yes":
            if doubleroot:
                X = 0.01
                equ_deriv = Differentiation(equ, X)
                Xn = X - (function(X) / equ_deriv)
                Aitken = str(Aitkens_process_Newton(equ, X, Xn, i=0))
                solution2_label.config(text=Aitken)
            else:
                Aitken = str(Aitkens_process_Newton(equ, x_initial, initial_point[0], i=0))
                solution2_label.config(text=Aitken)

    root = ThemedTk(theme="equilux")
    root.title("Newton-Raphson Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equation_label = ttk.Label(main_frame, text="Enter the equation in term of x:")
    equation_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry = ttk.Entry(main_frame, width=20)
    equation_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    initial_value_label = ttk.Label(main_frame, text="Enter the initial value of x:")
    initial_value_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    initial_value_entry = ttk.Entry(main_frame, width=20)
    initial_value_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    choice_label = ttk.Label(main_frame, text="If you want to continue with Newton-Raphson write 'continue'\nIf you want to use Aitken's process write 'Aitken'")
    choice_label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    choice_entry = ttk.Entry(main_frame, width=20)
    choice_entry.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=1, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    choice2_label = ttk.Label(main_frame, text="If you want to try Aitken's process write 'yes'")
    choice2_label.grid(column=0, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    choice2_entry = ttk.Entry(main_frame, width=20)
    choice2_entry.grid(column=1, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    solution2_label = ttk.Label(main_frame, text="")
    solution2_label.grid(column=1, row=6, padx=10, pady=10, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()

def Hally_Method():
    def calculate():
        equ=equation_entry.get()
        X=float(X_entry.get())
        function=equation(equ)
        equ_deriv1=Differentiation(equ,X)
        equ_deriv2=SecondDifferentiation(equ,X)
        if(equ_deriv1==0 and equ_deriv2==0):
            messagebox.showerror("Error", "Hally method failed")
            return None
        Xn=X-((function(X)*equ_deriv1))/((equ_deriv1**2)-(function(X)*equ_deriv2))
        i=0
        flag=True
        while flag and i < 1000:
            Xn=float(X-((function(X)*equ_deriv1))/((equ_deriv1**2)-(function(X)*equ_deriv2)))
            if Xn == X:
                flag = False
            else:
                print("At iteration [", i + 1, "]: ", Xn)
                X = Xn
                i += 1
        solution_label.config(text=f"The solution is Convergence to {Xn} at the {i}th iteration")

    root = ThemedTk(theme="equilux")
    root.title("Hally Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equation_label = ttk.Label(main_frame, text="Enter the equation in term of x:")
    equation_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry = ttk.Entry(main_frame, width=20)
    equation_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    X_label = ttk.Label(main_frame, text="Enter the initial value of x:")
    X_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    X_entry = ttk.Entry(main_frame, width=20)
    X_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    root.mainloop()
def FixedPointIteration_Method():
    def create_entries():
        for widget in entry_frame.winfo_children():
            widget.destroy()

        equ_entry.clear()

        try:
            n = int(n_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            return

        for i in range(n):
            ttk.Label(entry_frame, text=f"Enter equation {i+1} in terms of x:").grid(column=0, row=i, padx=10, pady=5, sticky=tk.W)
            equ_entry.append(ttk.Entry(entry_frame, width=30))
            equ_entry[-1].grid(column=1, row=i, padx=10, pady=5, sticky=tk.W)

    def calculate():
        initial_point = []
        i = 0
        try:
            n = int(n_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            return
        
        equ = [entry.get() for entry in equ_entry]
        try:
            X = float(x_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid initial value of x")
            return

        x_initial = X

        if n == 1 and abs(Differentiation(equ[0], X)) >= 1:
            messagebox.showerror("Error", "The solution is divergence")
        elif n == 1 and abs(Differentiation(equ[0], X)) < 1:
            function = equation(equ[0])
            xn = function(X)
            j = 0
            flag2 = True
            while flag2 and j < 1000:
                xn = float(function(X))
                if xn == X:
                    flag2 = False
                else:
                    initial_point.append(xn)
                    print(f"At iteration [{j+1}]: {xn}")
                    X = xn
                    j += 1
            Xn = initial_point[0]
            solution_label.config(text=f"The solution is Convergence to {xn} at the {j}th iteration")
            choice2 = choice2_entry.get()
            if choice2 == "yes":
                Aitken = str(Aitkens_Process_Fixed(function, x_initial, Xn, i=0))
                solution2_label.config(text=Aitken)
        elif n > 1:
            convergence = abs(Differentiation(equ[0], X))
            function = equation(equ[0])
            function_String = equ[0]
            for j in range(n):
                if abs(Differentiation(equ[j], X)) < convergence:
                    convergence = abs(Differentiation(equ[j], X))
                    function = equation(equ[j])
                    function_String = equ[j]
            best_label.config(text=f"The equation: {function_String} will give the best solution")
            xn = function(X)
            j = 0
            flag2 = True
            while flag2 and j < 1000:
                xn = float(function(X))
                if xn == X:
                    flag2 = False
                else:
                    initial_point.append(xn)
                    print(f"At iteration [{j+1}]: {xn}")
                    X = xn
                    j += 1
            Xn = initial_point[0]
            solution_label.config(text=f"The solution is Convergence to {xn} at the {j}th iteration")
            choice2 = choice2_entry.get()
            if choice2 == "yes":
                Aitken = str(Aitkens_Process_Fixed(function, x_initial, Xn, i=0))
                solution2_label.config(text=Aitken)

    root = ThemedTk(theme="equilux")
    root.title("Fixed Point Iteration Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    n_label = ttk.Label(main_frame, text="Enter number of equations:")
    n_label.grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)

    n_entry = ttk.Entry(main_frame, width=20)
    n_entry.grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)

    create_entries_button = ttk.Button(main_frame, text="Create Entries", command=create_entries)
    create_entries_button.grid(column=2, row=0, padx=10, pady=10, sticky=tk.W)

    entry_frame = ttk.Frame(main_frame, padding="10")
    entry_frame.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    x_label = ttk.Label(main_frame, text="Enter the initial value of x:")
    x_label.grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)

    x_entry = ttk.Entry(main_frame, width=20)
    x_entry.grid(column=1, row=2, padx=10, pady=10, sticky=tk.W)
    
    best_label = ttk.Label(main_frame, text="")
    best_label.grid(column=0, row=3, columnspan=3, padx=10, pady=10, sticky=tk.W)

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=4, padx=10, pady=10, sticky=tk.W)

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=0, row=5, columnspan=3, padx=10, pady=10, sticky=tk.W)

    choice2_label = ttk.Label(main_frame, text="If you want to try Aitken's process write 'yes'")
    choice2_label.grid(column=0, row=6, padx=10, pady=10, sticky=tk.W)

    choice2_entry = ttk.Entry(main_frame, width=20)
    choice2_entry.grid(column=1, row=6, padx=10, pady=10, sticky=tk.W)

    solution2_label = ttk.Label(main_frame, text="")
    solution2_label.grid(column=0, row=7, columnspan=3, padx=10, pady=10, sticky=tk.W)

    equ_entry = []

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()
def Secant_Method():
    def calculate():
        equ = equation_entry.get()
        x0 = float(x0_entry.get())
        x1 = float(x1_entry.get())
        function = equation(equ)
        if function(x0)*function(x1)>=0:
            messagebox.showerror("Error", "Secant method failed")
            return None
        Xn = x1 - function(x1) * (x1 - x0) / (function(x1) - function(x0))
        i = 0
        flag = True
        while flag and i < 1000:
            Xn = x1 - function(x1) * (x1 - x0) / (function(x1) - function(x0))
            if Xn == x1:
                flag = False
            else:
                print("At iteration [", i + 1, "]: ", Xn)
                x0 = x1
                x1 = Xn
                i += 1
        solution_label.config(text=f"The solution is Convergence to {Xn} at the {i}th iteration")

    root = ThemedTk(theme="equilux")
    root.title("Secant Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equation_label = ttk.Label(main_frame, text="Enter the equation in term of x:")
    equation_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry = ttk.Entry(main_frame, width=20)
    equation_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    x0_label = ttk.Label(main_frame, text="Enter the value of the first X:")
    x0_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    x0_entry = ttk.Entry(main_frame, width=20)
    x0_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    x1_label = ttk.Label(main_frame, text="Enter the value of the second X:")
    x1_label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    x1_entry = ttk.Entry(main_frame, width=20)
    x1_entry.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))


    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    root.mainloop()
def False_Position_Method():
    def calculate():
        equ = equation_entry.get()
        A = float(A_entry.get())
        B = float(B_entry.get())
        function = equation(equ)
        if function(A) * function(B) >= 0:
            messagebox.showerror("Error", "False position method failed")
            return None
        i = 0
        flag = True
        while flag and i < 1000:
            C = B - (function(B) * (B - A)) / (function(B) - function(A))
            i += 1
            if C == A or C == B:
                flag = False
            print("At iteration [", i, "]: ", C)
            if function(A) * function(C) >= 0:
                A = C
            else:
                B = C
        solution_label.config(text=f"The solution is Convergence to {C} at the {i}th iteration")

    root = ThemedTk(theme="equilux")
    root.title("False Position Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equation_label = ttk.Label(main_frame, text="Enter the equation in term of x:")
    equation_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry = ttk.Entry(main_frame, width=20)
    equation_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_label = ttk.Label(main_frame, text="Enter the value of A:")
    A_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the value of B:")
    B_label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()

def Jacobi_Method():
    def calculate():
        equ1=equation_entry1.get()
        equ2=equation_entry2.get()
        equ3=equation_entry3.get()
        function1=equationx1(isolatex1(equ1))
        function2=equationx2(isolatex2(equ2))
        function3=equationx3(isolatex3(equ3))
        boolean=test_coefficients(equ1,equ2,equ3)
        if(boolean==False):
            x1=float(x1_entry.get())
            x2=float(x2_entry.get())
            x3=float(x3_entry.get())
            x1new=function1(x2,x3)
            i=1
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
                x1new=function1(x2,x3)
                print("At iteration ",i," x1= ",x1new)
                x2new=function2(x1,x3)
                print("At iteration ",i," x2= ",x2new)
                x3new=function3(x1,x2)
                print("At iteration ",i," x3= ",x3new)
                if check_decimal_after_dot(x1new) and check_decimal_after_dot(x2new) and check_decimal_after_dot(x3new):
                    flag=False
            solution_label.config(text=f"The solution is Convergence to x1= {x1new}  x2= {x2new}  x3= {x3new} at the {i}th iteration")    
        else:
            messagebox.showerror("Error", "Jacobi method failed")
   

    

    root =ThemedTk(theme="equilux")
    root.title("Jacobi Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equation_label1 = ttk.Label(main_frame, text="Enter the first equation in term of x1,x2,x3,=:")
    equation_label1.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry1 = ttk.Entry(main_frame, width=20)
    equation_entry1.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_label2 = ttk.Label(main_frame, text="Enter the second equation in term of x1,x2,x3,=:")
    equation_label2.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry2 = ttk.Entry(main_frame, width=20)
    equation_entry2.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_label3 = ttk.Label(main_frame, text="Enter the third equation in term of x1,x2,x3,=:")
    equation_label3.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry3 = ttk.Entry(main_frame, width=20)
    equation_entry3.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    x1_label = ttk.Label(main_frame, text="Enter the initial value of x1:")
    x1_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    x1_entry = ttk.Entry(main_frame, width=20)
    x1_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    x2_label = ttk.Label(main_frame, text="Enter the initial value of x2:")
    x2_label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    x2_entry = ttk.Entry(main_frame, width=20)
    x2_entry.grid(column=1, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    x3_label = ttk.Label(main_frame, text="Enter the initial value of x3:")
    x3_label.grid(column=0, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    x3_entry = ttk.Entry(main_frame, width=20)
    x3_entry.grid(column=1, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=6, padx=10, pady=10, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=1, row=7, padx=10, pady=10, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    root.mainloop()

def Newton_forward():
    def create_entries():
        for entry in x_entries + y_entries:
            entry.destroy()
        x_entries.clear()
        y_entries.clear()

        try:
            size = int(size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size.")
            return

        for i in range(size):
            ttk.Label(entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_entries.append(ttk.Entry(entry_frame, width=20))
            x_entries[-1].grid(column=1, row=i)

            ttk.Label(entry_frame, text=f"Enter value of y{i+1}:").grid(column=2, row=i)
            y_entries.append(ttk.Entry(entry_frame, width=20))
            y_entries[-1].grid(column=3, row=i)
        
        interp_row_start = size
        ttk.Label(entry_frame, text="Enter the number of values of x-coordinates where interpolation is desired:").grid(column=0, row=interp_row_start + 1, columnspan=2)
        size_x_interp_entry.grid(column=2, row=interp_row_start + 1, padx=10, pady=10, sticky=(tk.W, tk.E))
        ttk.Button(entry_frame, text="Create Interpolation Entries", command=create_interp_entries).grid(column=0, row=interp_row_start + 2, columnspan=4)

    def create_interp_entries():
        for entry in x_interp_entries:
            entry.destroy()
        x_interp_entries.clear()
    
        try:
            size_x_interp = int(size_x_interp_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size of interpolation points.")
            return

        for i in range(size_x_interp):
            ttk.Label(interp_entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_interp_entries.append(ttk.Entry(interp_entry_frame, width=20))
            x_interp_entries[-1].grid(column=1, row=i)


    def calculate():
        try:
            size = int(size_entry.get())
            x = np.array([float(x_entries[i].get()) for i in range(size)])
            y = np.array([float(y_entries[i].get()) for i in range(size)])
            size_x_interp = int(size_x_interp_entry.get())
            x_interp = np.array([float(x_interp_entries[i].get()) for i in range(size_x_interp)])
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields.")
            return

        print("x:", x)
        print("y:", y)

        interval = x[1] - x[0]
        if not all(np.isclose(x[i + 1] - x[i], interval) for i in range(len(x) - 1)):
            messagebox.showerror("Error", "Newton Forward method failed as interval between x values is not constant")
            return None

        print("x_interp:", x_interp)
        n = len(x)
        h = x[1] - x[0]
        forward_diff = np.zeros((n, n))
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

        print("Polynomial equation: ", polynomial_eq)
        print("Simplified Polynomial equation before substitution:", expand(polynomial_eq))

        first_derivative = diff(polynomial_eq, p)
        second_derivative = diff(polynomial_eq, p, 2)

        interpolated_values_text = "Interpolated values:\n"
        first_derivative_values_text = "First derivative values:\n"
        second_derivative_values_text = "Second derivative values:\n"
        for i in range(size_x_interp):
            interpolated_values_text += f"At x = {x_interp[i]}: {polynomial_eq.subs(p, (x_interp[i] - x[0]) / h)}\n"
            first_derivative_values_text += f"At x = {x_interp[i]}: { (1 / h) * (first_derivative.subs(p, (x_interp[i] - x[0]) / h))}\n"
            second_derivative_values_text += f"At x = {x_interp[i]}: { (1 / math.pow(h, 2)) * (second_derivative.subs(p, (x_interp[i] - x[0]) / h))}\n"

        solution_label.config(text=interpolated_values_text)
        solution_label1.config(text=first_derivative_values_text)
        solution_label2.config(text=second_derivative_values_text)

        a = float(A_entry.get())
        b = float(B_entry.get())
        

        integral = integrate(polynomial_eq, (p, (a - x[0]) / h, (b - x[0]) / h))
        solution_label3.config(text=f"Integration: {h * integral}")



    root = ThemedTk(theme="equilux")
    root.title("Newton forward Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    size_label = ttk.Label(main_frame, text="Enter the number of values: ")
    size_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    size_entry = ttk.Entry(main_frame, width=20)
    size_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    create_entries_button = ttk.Button(main_frame, text="Create Entries", command=create_entries)
    create_entries_button.grid(column=2, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    entry_frame = ttk.Frame(main_frame, padding="10")
    entry_frame.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    x_entries = []
    y_entries = []
    x_interp_entries = []

    size_x_interp_entry = ttk.Entry(entry_frame, width=20)
    
    interp_entry_frame = ttk.Frame(main_frame, padding="10")
    interp_entry_frame.grid(column=0, row=2, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    A_label = ttk.Label(main_frame, text="Enter the lower limit of integration:")
    A_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the upper limit of integration:")
    B_label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=5, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=0, row=6, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label1 = ttk.Label(main_frame, text="")
    solution_label1.grid(column=0, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label2 = ttk.Label(main_frame, text="")
    solution_label2.grid(column=0, row=8, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label3 = ttk.Label(main_frame, text="")
    solution_label3.grid(column=0, row=9, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))



    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()

def Newton_backward():
    def create_entries():
        for entry in x_entries + y_entries:
            entry.destroy()
        x_entries.clear()
        y_entries.clear()

        try:
            size = int(size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size.")
            return

        for i in range(size):
            ttk.Label(entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_entries.append(ttk.Entry(entry_frame, width=20))
            x_entries[-1].grid(column=1, row=i)

            ttk.Label(entry_frame, text=f"Enter value of y{i+1}:").grid(column=2, row=i)
            y_entries.append(ttk.Entry(entry_frame, width=20))
            y_entries[-1].grid(column=3, row=i)
        
        interp_row_start = size
        ttk.Label(entry_frame, text="Enter the number of values of x-coordinates where interpolation is desired:").grid(column=0, row=interp_row_start + 1, columnspan=2)
        size_x_interp_entry.grid(column=2, row=interp_row_start + 1, padx=10, pady=10, sticky=(tk.W, tk.E))
        ttk.Button(entry_frame, text="Create Interpolation Entries", command=create_interp_entries).grid(column=0, row=interp_row_start + 2, columnspan=4)

    def create_interp_entries():
        for entry in x_interp_entries:
            entry.destroy()
        x_interp_entries.clear()
    
        try:
            size_x_interp = int(size_x_interp_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size of interpolation points.")
            return

        for i in range(size_x_interp):
            ttk.Label(interp_entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_interp_entries.append(ttk.Entry(interp_entry_frame, width=20))
            x_interp_entries[-1].grid(column=1, row=i)

    def calculate():
        try:
            size = int(size_entry.get())
            x = np.array([float(x_entries[i].get()) for i in range(size)])
            y = np.array([float(y_entries[i].get()) for i in range(size)])

            size_x_interp = int(size_x_interp_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields.")
            return
        x_interp = np.array([float(x_interp_entries[i].get()) for i in range(size_x_interp)])
        print("x:", x)
        print("y:", y)
        
        interval = x[1] - x[0]
        if not all(np.isclose(x[i + 1] - x[i], interval) for i in range(len(x) - 1)):
            messagebox.showerror("Error", "Newton backward method failed as interval between x values is not constant")
            return None

        print("x_interp:", x_interp)
        n = len(x)
        h = x[1] - x[0]
        backward_diff = np.zeros((n, n))
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
        print("Polynomial equation: ", polynomial_eq)
        print("Simplified Polynomial equation before substitution:", expand(polynomial_eq))
        first_derivative = diff(polynomial_eq, s)
        second_derivative = diff(polynomial_eq, s,2)
        interpolated_values_text = "Interpolated values:\n"
        first_derivative_values_text = "First derivative values:\n"
        second_derivative_values_text = "Second derivative values:\n"
        for i in range(size_x_interp):
            interpolated_values_text += f"At x = {x_interp[i]}: { polynomial_eq.subs(s, (x_interp[i] - x[-1]) / h)}\n"
            first_derivative_values_text += f"At x = {x_interp[i]}: { (1 / h) * (first_derivative.subs(s, (x_interp[i] - x[-1]) / h))}\n"
            second_derivative_values_text += f"At x = {x_interp[i]}: { (1 / math.pow(h, 2)) * (second_derivative.subs(s, (x_interp[i] - x[-1]) / h))}\n"

        solution_label.config(text=interpolated_values_text)
        solution_label1.config(text=first_derivative_values_text)
        solution_label2.config(text=second_derivative_values_text)
       

        a = float(A_entry.get())
        b = float(B_entry.get())
        integral = integrate(polynomial_eq, (s, (a-x[0])/h,(b-x[0])/h))
        solution_label3.config(text=f"Integration: {h * integral}")

    root = ThemedTk(theme="equilux")
    root.title("Newton Backward Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    size_label = ttk.Label(main_frame, text="Enter the number of values: ")
    size_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    size_entry = ttk.Entry(main_frame, width=20)
    size_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    create_entries_button = ttk.Button(main_frame, text="Create Entries", command=create_entries)
    create_entries_button.grid(column=2, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    entry_frame = ttk.Frame(main_frame, padding="10")
    entry_frame.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    x_entries = []
    y_entries = []
    x_interp_entries = []

    size_x_interp_entry = ttk.Entry(entry_frame, width=20)
    
    interp_entry_frame = ttk.Frame(main_frame, padding="10")
    interp_entry_frame.grid(column=0, row=2, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    A_label = ttk.Label(main_frame, text="Enter the lower limit of integration:")
    A_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the upper limit of integration:")
    B_label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=5, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=0, row=6, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label1 = ttk.Label(main_frame, text="")
    solution_label1.grid(column=0, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label2 = ttk.Label(main_frame, text="")
    solution_label2.grid(column=0, row=8, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label3 = ttk.Label(main_frame, text="")
    solution_label3.grid(column=0, row=9, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))


    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()

def Newton_divided_difference_interpolation():
    def create_entries():
        for entry in x_entries + y_entries:
            entry.destroy()
        x_entries.clear()
        y_entries.clear()

        try:
            size = int(size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size.")
            return

        for i in range(size):
            ttk.Label(entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_entries.append(ttk.Entry(entry_frame, width=20))
            x_entries[-1].grid(column=1, row=i)

            ttk.Label(entry_frame, text=f"Enter value of y{i+1}:").grid(column=2, row=i)
            y_entries.append(ttk.Entry(entry_frame, width=20))
            y_entries[-1].grid(column=3, row=i)
        
        interp_row_start = size
        ttk.Label(entry_frame, text="Enter the number of values of x-coordinates where interpolation is desired:").grid(column=0, row=interp_row_start + 1, columnspan=2)
        size_x_interp_entry.grid(column=2, row=interp_row_start + 1, padx=10, pady=10, sticky=(tk.W, tk.E))
        ttk.Button(entry_frame, text="Create Interpolation Entries", command=create_interp_entries).grid(column=0, row=interp_row_start + 2, columnspan=4)

    def create_interp_entries():
        for entry in x_interp_entries:
            entry.destroy()
        x_interp_entries.clear()
    
        try:
            size_x_interp = int(size_x_interp_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size of interpolation points.")
            return

        for i in range(size_x_interp):
            ttk.Label(interp_entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_interp_entries.append(ttk.Entry(interp_entry_frame, width=20))
            x_interp_entries[-1].grid(column=1, row=i)
    def calculate():
        try:
            size = int(size_entry.get())
            x = np.array([float(x_entries[i].get()) for i in range(size)])
            y = np.array([float(y_entries[i].get()) for i in range(size)])
            size_x_interp = int(size_x_interp_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields.")
            return
        print("x:", x)
        print("y:", y)
        x_interp = np.array([float(x_interp_entries[i].get()) for i in range(size_x_interp)])
        print("x_interp:", x_interp)
        n = len(x)
        t = symbols('t')
        F = [[0 for _ in range(n)] for _ in range(n)]
        for j in range(n):
            F[j][0] = y[j]
        for j in range(1, n):
            for i in range(n - j):
                F[i][j] = (F[i+1][j-1] - F[i][j-1]) / (x[i+j] - x[i])
        polynomial_eq = F[0][0]
        for j in range(1, n):
            term = 1
            for k in range(j):
                term *= (t - x[k])
            polynomial_eq += F[0][j] * term
        print("Polynomial equation before substitution:", polynomial_eq)
        print("Simplified Polynomial equation before substitution:", expand(polynomial_eq))
        first_derivative = diff(polynomial_eq, t)
        second_derivative = diff(polynomial_eq, t,2)
        interpolated_values_text = "Interpolated values:\n"
        first_derivative_values_text = "First derivative values:\n"
        second_derivative_values_text = "Second derivative values:\n"
        for i in range(size_x_interp):
            interpolated_values_text += f"At x = {x_interp[i]}: {polynomial_eq.subs(t, x_interp[i])}\n"
            first_derivative_values_text += f"At x = {x_interp[i]}: {first_derivative.subs(t,(x_interp[i]))}\n"
            second_derivative_values_text += f"At x = {x_interp[i]}: {second_derivative.subs(t,(x_interp[i]))}\n"

        solution_label.config(text=interpolated_values_text)
        solution_label1.config(text=first_derivative_values_text)
        solution_label2.config(text=second_derivative_values_text)
        a = float(A_entry.get())
        b = float(B_entry.get())
        integral = integrate(polynomial_eq, (t, a,b))
        solution_label3.config(text=f"Integration :{integral}")

    root = ThemedTk(theme="equilux")
    root.title("Newton divided difference interpolation Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    size_label = ttk.Label(main_frame, text="Enter the number of values: ")
    size_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    size_entry = ttk.Entry(main_frame, width=20)
    size_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    create_entries_button = ttk.Button(main_frame, text="Create Entries", command=create_entries)
    create_entries_button.grid(column=2, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    entry_frame = ttk.Frame(main_frame, padding="10")
    entry_frame.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    x_entries = []
    y_entries = []
    x_interp_entries = []

    size_x_interp_entry = ttk.Entry(entry_frame, width=20)
    
    interp_entry_frame = ttk.Frame(main_frame, padding="10")
    interp_entry_frame.grid(column=0, row=2, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    A_label = ttk.Label(main_frame, text="Enter the lower limit of integration:")
    A_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the upper limit of integration:")
    B_label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=5, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=0, row=6, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label1 = ttk.Label(main_frame, text="")
    solution_label1.grid(column=0, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label2 = ttk.Label(main_frame, text="")
    solution_label2.grid(column=0, row=8, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label3 = ttk.Label(main_frame, text="")
    solution_label3.grid(column=0, row=9, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()

def LaGrange():
    def create_entries():
        for entry in x_entries + y_entries:
            entry.destroy()
        x_entries.clear()
        y_entries.clear()

        try:
            size = int(size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size.")
            return

        for i in range(size):
            ttk.Label(entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_entries.append(ttk.Entry(entry_frame, width=20))
            x_entries[-1].grid(column=1, row=i)

            ttk.Label(entry_frame, text=f"Enter value of y{i+1}:").grid(column=2, row=i)
            y_entries.append(ttk.Entry(entry_frame, width=20))
            y_entries[-1].grid(column=3, row=i)
        
        interp_row_start = size
        ttk.Label(entry_frame, text="Enter the number of values of x-coordinates where interpolation is desired:").grid(column=0, row=interp_row_start + 1, columnspan=2)
        size_x_interp_entry.grid(column=2, row=interp_row_start + 1, padx=10, pady=10, sticky=(tk.W, tk.E))
        ttk.Button(entry_frame, text="Create Interpolation Entries", command=create_interp_entries).grid(column=0, row=interp_row_start + 2, columnspan=4)

    def create_interp_entries():
        for entry in x_interp_entries:
            entry.destroy()
        x_interp_entries.clear()
    
        try:
            size_x_interp = int(size_x_interp_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size of interpolation points.")
            return

        for i in range(size_x_interp):
            ttk.Label(interp_entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_interp_entries.append(ttk.Entry(interp_entry_frame, width=20))
            x_interp_entries[-1].grid(column=1, row=i)
    def calculate():
        try:
            size = int(size_entry.get())
            x = np.array([float(x_entries[i].get()) for i in range(size)])
            y = np.array([float(y_entries[i].get()) for i in range(size)])
            size_x_interp = int(size_x_interp_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields.")
            return
        print("x:", x)
        print("y:", y)
        x_interp = np.array([float(x_interp_entries[i].get()) for i in range(size_x_interp)])
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
        print("Polynomial equation: ", polynomial_eq)
        print("Simplified Polynomial equation before substitution:", expand(polynomial_eq))
        first_derivative = diff(polynomial_eq, t)
        second_derivative = diff(polynomial_eq, t, 2)

        interpolated_values_text = "Interpolated values:\n"
        first_derivative_values_text = "First derivative values:\n"
        second_derivative_values_text = "Second derivative values:\n"
        for i in range(size_x_interp):
            interpolated_values_text += f"At x = {x_interp[i]}: {polynomial_eq.subs(t, x_interp[i])}\n"
            first_derivative_values_text += f"At x = {x_interp[i]}: {first_derivative.subs(t, x_interp[i])}\n"
            second_derivative_values_text += f"At x = {x_interp[i]}: {second_derivative.subs(t, x_interp[i])}\n"

        solution_label.config(text=interpolated_values_text)
        solution_label1.config(text=first_derivative_values_text)
        solution_label2.config(text=second_derivative_values_text)

        a = float(A_entry.get())
        b = float(B_entry.get())
        integral = integrate(polynomial_eq, (t, a, b))
        solution_label3.config(text=f"Integration from {a} to {b}: {integral}")
       

    root = ThemedTk(theme="equilux")
    root.title("La Grange Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    size_label = ttk.Label(main_frame, text="Enter the number of values: ")
    size_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    size_entry = ttk.Entry(main_frame, width=20)
    size_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    create_entries_button = ttk.Button(main_frame, text="Create Entries", command=create_entries)
    create_entries_button.grid(column=2, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    entry_frame = ttk.Frame(main_frame, padding="10")
    entry_frame.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    x_entries = []
    y_entries = []
    x_interp_entries = []

    size_x_interp_entry = ttk.Entry(entry_frame, width=20)
    
    interp_entry_frame = ttk.Frame(main_frame, padding="10")
    interp_entry_frame.grid(column=0, row=2, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    A_label = ttk.Label(main_frame, text="Enter the lower limit of integration:")
    A_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the upper limit of integration:")
    B_label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=5, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=0, row=6, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label1 = ttk.Label(main_frame, text="")
    solution_label1.grid(column=0, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label2 = ttk.Label(main_frame, text="")
    solution_label2.grid(column=0, row=8, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label3 = ttk.Label(main_frame, text="")
    solution_label3.grid(column=0, row=9, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()

def Trapezoidal():
    def calculate():
    # n: number of values in table -1
        n=int(N_entry.get())
        equ=equation_entry.get()
        function=equation(equ)
        a = float(A_entry.get())
        b = float(B_entry.get())
        h=(b-a)/n
        print ("h=",h)
        x = np.linspace(a, b, n+1)
        y = function(x)
        print("X:",x)
        print("Y:",y)
        integral_T = h * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]))
        solution_label.config(text=f"Trapezoidal Integral= {integral_T}")
        exact_Integral=Integration(equ,a,b)
        solution_label1.config(text=f"Exact Integral= {exact_Integral}")
        absolute_error=abs(exact_Integral-integral_T)
        relative_error=abs(absolute_error/exact_Integral)
        solution_label2.config(text=f"Absolute Error: {'{:.20f}'.format(absolute_error)}")
        solution_label3.config(text=f"Relative Error: {'{:.20f}'.format(relative_error)}")
        sec_Differentiation=SecondDifferentiation(equ,a)
        tr_T= ((h**2*(b-a))/12)*sec_Differentiation
        solution_label4.config(text=f"Tranction Error of Trapezoidal : {'{:.20f}'.format(abs(tr_T))}")
        if(sec_Differentiation<0):
            solution_label5.config(text=f"{integral_T} <=  Exact integration <= {(integral_T+abs(tr_T))}")
        else:
            solution_label5.config(text=f"{(integral_T-abs(tr_T))} <=  Exact integration <= {integral_T}")
    
    root = ThemedTk(theme="equilux")
    root.title("Trapezoidal Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    N_label = ttk.Label(main_frame, text="Enter the N:")
    N_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    N_entry = ttk.Entry(main_frame, width=20)
    N_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_label = ttk.Label(main_frame, text="Enter the equation in term of x:")
    equation_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry = ttk.Entry(main_frame, width=20)
    equation_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    label = ttk.Label(main_frame, text="Enter boundaries:")
    label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_label = ttk.Label(main_frame, text="Enter the value of A:")
    A_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the value of B:")
    B_label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=5, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=0, row=6, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label1 = ttk.Label(main_frame, text="")
    solution_label1.grid(column=3, row=6, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label2 = ttk.Label(main_frame, text="")
    solution_label2.grid(column=0, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label3 = ttk.Label(main_frame, text="")
    solution_label3.grid(column=3, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label4 = ttk.Label(main_frame, text="")
    solution_label4.grid(column=0, row=8, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label5 = ttk.Label(main_frame, text="")
    solution_label5.grid(column=3, row=8, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()

def Simpson():
    def calculate():
    # n: number of values in table -1
        n=int(N_entry.get())
        equ=equation_entry.get()
        function=equation(equ)
        a = float(A_entry.get())
        b = float(B_entry.get())
        h=(b-a)/n
        print ("h=",h)
        x = np.linspace(a, b, n+1)
        y = function(x)
        print("X:",x)
        print("Y:",y)
        integral_S = h / 3 * (y[0] + y[-1] + 2 * np.sum(y[2:-2:2]) + 4 * np.sum(y[1:-1:2]))
        solution_label.config(text=f"Simpson's Integral= {integral_S}")
        exact_Integral=Integration(equ,a,b)
        solution_label1.config(text=f"Exact Integral= {exact_Integral}")
        absolute_error=abs(exact_Integral-integral_S)
        relative_error=abs(absolute_error/exact_Integral)
        solution_label2.config(text=f"Absolute Error: {'{:.20f}'.format(absolute_error)}")
        solution_label3.config(text=f"Relative Error: {'{:.20f}'.format(relative_error)}")
        fourth_differentiation=FourthDifferentiation(equ,a)
        tr_S= ((h**4*(b-a))/180)*fourth_differentiation
        solution_label4.config(text=f"Tranction Error of Simpson : {'{:.20f}'.format(abs(tr_S))}")
        if(fourth_differentiation<0):
            solution_label5.config(text=f"{integral_S} <=  Exact integration <= {(integral_S+abs(tr_S))}")
        else:
            solution_label5.config(text=f"{(integral_S-abs(tr_S))} <=  Exact integration <= {integral_S}")
    
    
    root = ThemedTk(theme="equilux")
    root.title("Simpson Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    N_label = ttk.Label(main_frame, text="Enter the N:")
    N_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    N_entry = ttk.Entry(main_frame, width=20)
    N_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_label = ttk.Label(main_frame, text="Enter the equation in term of x:")
    equation_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry = ttk.Entry(main_frame, width=20)
    equation_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    label = ttk.Label(main_frame, text="Enter boundaries:")
    label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_label = ttk.Label(main_frame, text="Enter the value of A:")
    A_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the value of B:")
    B_label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=5, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=0, row=6, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label1 = ttk.Label(main_frame, text="")
    solution_label1.grid(column=3, row=6, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label2 = ttk.Label(main_frame, text="")
    solution_label2.grid(column=0, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label3 = ttk.Label(main_frame, text="")
    solution_label3.grid(column=3, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label4 = ttk.Label(main_frame, text="")
    solution_label4.grid(column=0, row=8, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label5 = ttk.Label(main_frame, text="")
    solution_label5.grid(column=3, row=8, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()

def Gauss_Seidel_Method():
    def calculate():
        equ1 = equ1_entry.get()
        equ2 = equ2_entry.get()
        equ3 = equ3_entry.get()
        function1 = equationx1(isolatex1(equ1))
        function2 = equationx2(isolatex2(equ2))
        function3 = equationx3(isolatex3(equ3))
        x1 = float(x1_entry.get())
        x2 = float(x2_entry.get())
        x3 = float(x3_entry.get())
        i = 0
        flag = True
        while flag and i < 1000:
            i += 1
            x1 = function1(x2, x3)
            print("At iteration ",i," x1= ",x1)
            x2 = function2(x1, x3)
            print("At iteration ",i," x2= ",x2)
            x3 = function3(x1, x2)
            print("At iteration ",i," x3= ",x3)
            if check_decimal_after_dot(x1) and check_decimal_after_dot(x2) and check_decimal_after_dot(x3):
                flag = False
        solution_label.config(text=f"The solution is Convergence to x1= {x1}  x2= {x2}  x3= {x3} at the {i}th iteration")

    root = ThemedTk(theme="equilux")
    root.title("Gauss-Seidel Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equ1_label = ttk.Label(main_frame, text="Enter the first equation in term of x1,x2,x3,=:")
    equ1_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equ1_entry = ttk.Entry(main_frame, width=20)
    equ1_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equ2_label = ttk.Label(main_frame, text="Enter the second equation in term of x1,x2,x3,=:")
    equ2_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    equ2_entry = ttk.Entry(main_frame, width=20)
    equ2_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    equ3_label = ttk.Label(main_frame, text="Enter the third equation in term of x1,x2,x3,=:")
    equ3_label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    equ3_entry = ttk.Entry(main_frame, width=20)
    equ3_entry.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    x1_label = ttk.Label(main_frame, text="Enter the initial value of x1:")
    x1_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    x1_entry = ttk.Entry(main_frame, width=20)
    x1_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    x2_label = ttk.Label(main_frame, text="Enter the initial value of x2:")
    x2_label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    x2_entry = ttk.Entry(main_frame, width=20)
    x2_entry.grid(column=1, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    x3_label = ttk.Label(main_frame, text="Enter the initial value of x3:")
    x3_label.grid(column=0, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    x3_entry = ttk.Entry(main_frame, width=20)
    x3_entry.grid(column=1, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=6, padx=10, pady=10, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=1, row=7, padx=10, pady=10, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    root.mainloop()


def regression():
    def create_entries():
        for entry in x_entries + y_entries:
            entry.destroy()
        x_entries.clear()
        y_entries.clear()

        try:
            n = int(n_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size.")
            return

        for i in range(n):
            ttk.Label(entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_entries.append(ttk.Entry(entry_frame, width=20))
            x_entries[-1].grid(column=1, row=i)

            ttk.Label(entry_frame, text=f"Enter value of y{i+1}:").grid(column=2, row=i)
            y_entries.append(ttk.Entry(entry_frame, width=20))
            y_entries[-1].grid(column=3, row=i)
    def calculate():
        n=int(n_entry.get())
        x = [float(x_entry.get()) for x_entry in x_entries]
        y = [float(y_entry.get()) for y_entry in y_entries]
        sumX = sum(x)
        sumY = sum(y)
        sumXsq = sum([x**2 for x in x])
        sumXY = sum([x*y for x, y in zip(x, y)])
        a = (sumY * sumXsq - sumX * sumXY) / (n * sumXsq - sumX**2)
        b = (n * sumXY - sumX * sumY) / (n * sumXsq - sumX**2)
        print("x list:", x)
        print("y list:", y)
        print("sum X:", sumX)
        print("sum y:", sumY)
        print("sum x^2:", sumXsq)
        print("sum X*Y:", sumXY)
        print("Equation1= ",n,"a+",sumX,"b=",sumY)
        print("Equation2= ",sumX,"a+",sumXsq,"b=",sumXY)
        solution_label.config(text=f"The regression line is y = {b} x + {a}")

    root = ThemedTk(theme="equilux")
    root.title("Regression Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    n_label = ttk.Label(main_frame, text="Enter the number of data points: ")
    n_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    n_entry = ttk.Entry(main_frame, width=10)
    n_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    entry_frame = ttk.Frame(main_frame, padding="10")
    entry_frame.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    create_entries_button = ttk.Button(main_frame, text="Create Entries", command=create_entries)
    create_entries_button.grid(column=2, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))


    x_entries = []
    y_entries = []

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))


    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    root.mainloop()
def equationx(equ):
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    equation = sp.sympify(equ)
    return equation
def equationmu(equ):
    mu1, mu2, mu3 = sp.symbols('mu1 mu2 mu3')
    equation = sp.sympify(equ)
    return equation
def Jacobi_Method_Non_Linear_Equations():
    def calculate():
        equ1=equation_entry1.get()
        equ2=equation_entry2.get()
        equ3=equation_entry3.get()
        x1, x2, x3 = sp.symbols('x1 x2 x3')
        x11=float(x1_entry.get())
        x22=float(x2_entry.get())
        x33=float(x3_entry.get())
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
            solution_label.config(text=f'Matrix of New X: {x_new}')
            x11=solutions[0]
            x22=solutions[1]
            x33=solutions[2]
            x_new=matrixXs
    root =ThemedTk(theme="equilux")
    root.title("Jacobi Method Non Linear Equations")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equation_label1 = ttk.Label(main_frame, text="Enter the equation1 in term of x1,x2,x3: ")
    equation_label1.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry1 = ttk.Entry(main_frame, width=20)
    equation_entry1.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_label2 = ttk.Label(main_frame, text="Enter the equation1 in term of x1,x2,x3: ")
    equation_label2.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry2 = ttk.Entry(main_frame, width=20)
    equation_entry2.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_label3 = ttk.Label(main_frame, text="Enter the equation1 in term of x1,x2,x3: ")
    equation_label3.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry3 = ttk.Entry(main_frame, width=20)
    equation_entry3.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    x1_label = ttk.Label(main_frame, text="Enter the initial value of x1:")
    x1_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    x1_entry = ttk.Entry(main_frame, width=20)
    x1_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    x2_label = ttk.Label(main_frame, text="Enter the initial value of x2:")
    x2_label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    x2_entry = ttk.Entry(main_frame, width=20)
    x2_entry.grid(column=1, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    x3_label = ttk.Label(main_frame, text="Enter the initial value of x3:")
    x3_label.grid(column=0, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    x3_entry = ttk.Entry(main_frame, width=20)
    x3_entry.grid(column=1, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))
    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=6, padx=10, pady=10, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="")
    solution_label.grid(column=1, row=7, padx=10, pady=10, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    root.mainloop()

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
    def create_entries():
        for entry in x_entries + y_entries:
            entry.destroy()
        x_entries.clear()
        y_entries.clear()

        try:
            size = int(size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size.")
            return

        for i in range(size):
            ttk.Label(entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_entries.append(ttk.Entry(entry_frame, width=20))
            x_entries[-1].grid(column=1, row=i)

            ttk.Label(entry_frame, text=f"Enter value of y{i+1}:").grid(column=2, row=i)
            y_entries.append(ttk.Entry(entry_frame, width=20))
            y_entries[-1].grid(column=3, row=i)
    def calculate():
        try:
            size = int(size_entry.get())
            x = np.array([float(x_entries[i].get()) for i in range(size)])
            y = np.array([float(y_entries[i].get()) for i in range(size)])
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields.")
            return
        print("x:", x)
        print("y:", y)
        h= float(h_entry.get())
        forward_diff = forward_difference1(x, y, h)
        backward_diff = backward_difference1(x, y, h)  
        forward_diff_str = "Forward Differences:\n" + "\n".join([f"x={point[0]}, {point[1]}: {point[2]}" for point in forward_diff])
        forward_label.config(text=forward_diff_str)
        backward_diff_str = "Backward Differences:\n" + "\n".join([f"x={point[0]}, {point[1]}: {point[2]}" for point in backward_diff])
        backward_label.config(text=backward_diff_str)
    root = ThemedTk(theme="equilux")
    root.title("Two points Formula Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    size_label = ttk.Label(main_frame, text="Enter the number of values: ")
    size_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    size_entry = ttk.Entry(main_frame, width=20)
    size_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    create_entries_button = ttk.Button(main_frame, text="Create Entries", command=create_entries)
    create_entries_button.grid(column=2, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    entry_frame = ttk.Frame(main_frame, padding="10")
    entry_frame.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    x_entries = []
    y_entries = []
    
    h_label = ttk.Label(main_frame, text="Enter the value of h: ")
    h_label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    h_entry = ttk.Entry(main_frame, width=20)
    h_entry.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=3, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    forward_label = ttk.Label(main_frame, text="")
    forward_label.grid(column=0, row=4, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    backward_label = ttk.Label(main_frame, text="")
    backward_label.grid(column=0, row=5, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()

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
    def create_entries():
        for entry in x_entries + y_entries:
            entry.destroy()
        x_entries.clear()
        y_entries.clear()
        try:
            size = int(size_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for size.")
            return
        for i in range(size):
            ttk.Label(entry_frame, text=f"Enter value of X{i+1}:").grid(column=0, row=i)
            x_entries.append(ttk.Entry(entry_frame, width=20))
            x_entries[-1].grid(column=1, row=i)

            ttk.Label(entry_frame, text=f"Enter value of y{i+1}:").grid(column=2, row=i)
            y_entries.append(ttk.Entry(entry_frame, width=20))
            y_entries[-1].grid(column=3, row=i)
    def calculate():
        try:
            size = int(size_entry.get())
            x = np.array([float(x_entries[i].get()) for i in range(size)])
            y = np.array([float(y_entries[i].get()) for i in range(size)])
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers in all fields.")
            return 
        print("x:", x)
        print("y:", y)
        h= float(h_entry.get())
        forward_diff = forward_difference(x, y, h)
        backward_diff = backward_difference(x, y, h)
        central_diff = central_difference(x, y, h)
        forward_diff_str = "Forward Differences:\n" + "\n".join([f"x={point[0]}, {point[1]}: {point[2]}" for point in forward_diff])
        forward_label.config(text=forward_diff_str)
        central_diff_str = "Central Differences:\n" + "\n".join([f"x={point[0]}, {point[1]}: {point[2]}" for point in central_diff])
        central_label.config(text=central_diff_str)
        backward_diff_str = "Backward Differences:\n" + "\n".join([f"x={point[0]}, {point[1]}: {point[2]}" for point in backward_diff])
        backward_label.config(text=backward_diff_str)
    root = ThemedTk(theme="equilux")
    root.title("Three points Formula Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    size_label = ttk.Label(main_frame, text="Enter the number of values: ")
    size_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    size_entry = ttk.Entry(main_frame, width=20)
    size_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    create_entries_button = ttk.Button(main_frame, text="Create Entries", command=create_entries)
    create_entries_button.grid(column=2, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    entry_frame = ttk.Frame(main_frame, padding="10")
    entry_frame.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

    x_entries = []
    y_entries = []
    
    h_label = ttk.Label(main_frame, text="Enter the value of h: ")
    h_label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    h_entry = ttk.Entry(main_frame, width=20)
    h_entry.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=3, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    forward_label = ttk.Label(main_frame, text="")
    forward_label.grid(column=0, row=4, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    central_label = ttk.Label(main_frame, text="")
    central_label.grid(column=0, row=5, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    backward_label = ttk.Label(main_frame, text="")
    backward_label.grid(column=0, row=6, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()
def Runge_kutta_3():
    def calculate():
        equ=equation_entry.get()
        function=equation_x_y(equ)
        x0=float(x0_entry.get())
        y0=float(y0_entry.get())
        n=int(n_entry.get())
        print("Enter boundaries")
        a=float(A_entry.get())
        b=float(B_entry.get())
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
        result = "\n".join([f"x = {x}, y = {y}" for x, y in zip(x_values, y_values)])
        solution_label.config(text=result)

    root = ThemedTk(theme="equilux")
    root.title("Runge Kutta 3 Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equation_label = ttk.Label(main_frame, text="Enter the differential equation in term of x and y: ")
    equation_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry = ttk.Entry(main_frame, width=20)
    equation_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    x0_label = ttk.Label(main_frame, text="Enter the value of initial x:")
    x0_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    x0_entry = ttk.Entry(main_frame, width=20)
    x0_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    y0_label = ttk.Label(main_frame, text="Enter the value of initial y: ")
    y0_label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    y0_entry = ttk.Entry(main_frame, width=20)
    y0_entry.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    n_label = ttk.Label(main_frame, text="Enter number of iterations (n):")
    n_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    n_entry = ttk.Entry(main_frame, width=20)
    n_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    label = ttk.Label(main_frame, text="Enter boundaries:")
    label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_label = ttk.Label(main_frame, text="Enter the value of lower limit: ")
    A_label.grid(column=0, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the value of upper limit: ")
    B_label.grid(column=0, row=6, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=6, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="", justify=tk.LEFT)
    solution_label.grid(column=0, row=8, columnspan=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()
def Euler():
    def calculate():
        equ=equation_entry.get()
        function=equation_x_y(equ)
        x0=float(x0_entry.get())
        y0=float(y0_entry.get())
        n=int(n_entry.get())
        print("Enter boundaries")
        a=float(A_entry.get())
        b=float(B_entry.get())
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
        result = "\n".join([f"x = {x}, y = {y}" for x, y in zip(x_values, y_values)])
        solution_label.config(text=result)
    root = ThemedTk(theme="equilux")
    root.title("Euler Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equation_label = ttk.Label(main_frame, text="Enter the differential equation in term of x and y: ")
    equation_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry = ttk.Entry(main_frame, width=20)
    equation_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    x0_label = ttk.Label(main_frame, text="Enter the value of initial x:")
    x0_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    x0_entry = ttk.Entry(main_frame, width=20)
    x0_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    y0_label = ttk.Label(main_frame, text="Enter the value of initial y: ")
    y0_label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    y0_entry = ttk.Entry(main_frame, width=20)
    y0_entry.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    n_label = ttk.Label(main_frame, text="Enter number of iterations (n):")
    n_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    n_entry = ttk.Entry(main_frame, width=20)
    n_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    label = ttk.Label(main_frame, text="Enter boundaries:")
    label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_label = ttk.Label(main_frame, text="Enter the value of lower limit: ")
    A_label.grid(column=0, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the value of upper limit: ")
    B_label.grid(column=0, row=6, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=6, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="", justify=tk.LEFT)
    solution_label.grid(column=0, row=8, columnspan=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop()

def Euler_Modified():
    def calculate():
        equ=equation_entry.get()
        function=equation_x_y(equ)
        x0=float(x0_entry.get())
        y0=float(y0_entry.get())
        n=int(n_entry.get())
        print("Enter boundaries")
        a=float(A_entry.get())
        b=float(B_entry.get())
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
        result = "\n".join([f"x = {x}, y = {y}" for x, y in zip(x_values, y_values)])
        solution_label.config(text=result)
    root = ThemedTk(theme="equilux")
    root.title("Euler modified Method")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    equation_label = ttk.Label(main_frame, text="Enter the differential equation in term of x and y: ")
    equation_label.grid(column=0, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    equation_entry = ttk.Entry(main_frame, width=20)
    equation_entry.grid(column=1, row=0, padx=10, pady=10, sticky=(tk.W, tk.E))

    x0_label = ttk.Label(main_frame, text="Enter the value of initial x:")
    x0_label.grid(column=0, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    x0_entry = ttk.Entry(main_frame, width=20)
    x0_entry.grid(column=1, row=1, padx=10, pady=10, sticky=(tk.W, tk.E))

    y0_label = ttk.Label(main_frame, text="Enter the value of initial y: ")
    y0_label.grid(column=0, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    y0_entry = ttk.Entry(main_frame, width=20)
    y0_entry.grid(column=1, row=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    n_label = ttk.Label(main_frame, text="Enter number of iterations (n):")
    n_label.grid(column=0, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    n_entry = ttk.Entry(main_frame, width=20)
    n_entry.grid(column=1, row=3, padx=10, pady=10, sticky=(tk.W, tk.E))

    label = ttk.Label(main_frame, text="Enter boundaries:")
    label.grid(column=0, row=4, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_label = ttk.Label(main_frame, text="Enter the value of lower limit: ")
    A_label.grid(column=0, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    A_entry = ttk.Entry(main_frame, width=20)
    A_entry.grid(column=1, row=5, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_label = ttk.Label(main_frame, text="Enter the value of upper limit: ")
    B_label.grid(column=0, row=6, padx=10, pady=10, sticky=(tk.W, tk.E))

    B_entry = ttk.Entry(main_frame, width=20)
    B_entry.grid(column=1, row=6, padx=10, pady=10, sticky=(tk.W, tk.E))

    calculate_button = ttk.Button(main_frame, text="Calculate", command=calculate)
    calculate_button.grid(column=0, row=7, padx=10, pady=10, columnspan=3, sticky=(tk.W, tk.E))

    solution_label = ttk.Label(main_frame, text="", justify=tk.LEFT)
    solution_label.grid(column=0, row=8, columnspan=2, padx=10, pady=10, sticky=(tk.W, tk.E))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)
    root.mainloop() 
def GUI():
    window = ThemedTk(theme="equilux")  
    window.title("Numerical Project")
    Icon=tk.PhotoImage(file='Aiu.png')
    window.iconphoto(True,Icon)
    window.geometry("500x300")  

    
    style = ttk.Style(window)
    style.configure('TLabel', background='#333', foreground='#FFF', font=('Helvetica', 12))
    style.configure('TEntry', foreground='black', font=('Helvetica', 12))
    style.configure('TButton', background='#333', foreground='white', font=('Helvetica', 12))
    style.map('TButton', background=[('active', '#666')])

   
    padding_frame = ttk.Frame(window, padding="30")
    padding_frame.pack(expand=True, fill=tk.BOTH)

    label = ttk.Label(padding_frame, text="Select a Numerical Method to Run:", font=('Helvetica', 14, 'bold'))
    label.pack(pady=10)

  
    methods = ["bisection_method", "newton_raphson_method", "hally_method", "fixed_point_iteration_method", "secant_method",
              "false_position_method", "jacobi_method", "gauss_seidel_method","regression method","Jacobi_Method_Non_Linear_Equations",
              "La Grange","Newton_forward","Newton_backward","Newton_divided_difference_interpolation","Trapezoidal","Simpson",
              "Two_points_Formula","Three_points_Formula","Runge_kutta_3","Euler","Euler_Modified"]
    method_combobox = ttk.Combobox(padding_frame, values=methods, state="readonly", width=40, font=('Helvetica', 12))
    method_combobox.pack(pady=20)

 
    run_button = ttk.Button(padding_frame, text="Run Method", command=lambda: run(method_combobox.get()))
    run_button.pack(pady=10)

    window.mainloop()
def run(method):
    if method == "bisection_method":
        bisection_method()
    elif method == "newton_raphson_method":
        NewtonRaphson_Method()
    elif method == "hally_method":
        Hally_Method()
    elif method == "fixed_point_iteration_method":
        FixedPointIteration_Method()
    elif method == "secant_method":
        Secant_Method()
    elif method == "false_position_method":
       False_Position_Method()
    elif method == "jacobi_method":
        Jacobi_Method()
    elif method =="regression method":
        regression()
    elif method == "gauss_seidel_method":
        Gauss_Seidel_Method()
    elif method=="Jacobi_Method_Non_Linear_Equations":
        Jacobi_Method_Non_Linear_Equations()
    elif method=="Newton_forward":
        Newton_forward()
    elif method=="Newton_backward":
        Newton_backward()
    elif method=="Newton_divided_difference_interpolation":
        Newton_divided_difference_interpolation()
    elif method=="La Grange":
        LaGrange()
    elif method=="Trapezoidal":
        Trapezoidal()
    elif method=="Simpson":
        Simpson()
    elif method=="Two_points_Formula":
        Two_points_Formula()
    elif method=="Three_points_Formula":
        Three_points_Formula()
    elif method=="Runge_kutta_3":
        Runge_kutta_3()
    elif method=="Euler":
        Euler()
    elif method=="Euler_Modified":
        Euler_Modified()
    else:
        print("Invalid method")
GUI()

