import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math

t = sp.symbols('t')
r1 = sp.Matrix([sp.ln(t),sp.exp(-t),t**3])
r2 = sp.Matrix([2*sp.cos(sp.pi*t),2*sp.sin(sp.pi*t),3*t])

v1 = r1.diff(t).subs(t,2).evalf()
v2 = r2.diff(t).subs(t,1/3).evalf()

print("Tanget to r2 will be:" + "\n " + "[" + "\n ".join(f"{x:.5f}" for x in v1) + "]")
print("Tanget to r2 will be:" + "\n "+ "[" + "\n ".join(f"{x:.5f}" for x in v2) +"]")

#separator.join(iterable_of_strings) Here the separator can be \n or , ... followed by a iterable of strings.

# # Example 1: Basic join with a list of strings
# words = ["Hello", "World"]
# result = " ".join(words)
# print(result)  # Output: "Hello World"

# Finding vector parallel to the intersection of the two planes

n1 = np.array([3,-6,-2])
n2 = np.array([2,1,-2])
v = np.cross(n1,n2)
print(f"Vector parallel to the line of intersection is: {v}")

# Finding the velocity and acceleration of the vector r(t)
r3 = sp.Matrix([3*t,sp.sin(t),t**2])
v3 = r3.diff(t)
a3 = v3.diff(t)

theta = sp.acos(v3.dot(a3)/(v3.norm()*a3.norm()))
theta_func = sp.lambdify(t,theta,'numpy') 

#lambdify(args, expr, modules='numpy')
#converts symbolic expressions into fast, callable Python functions for numerical computation.

t_vals = np.linspace(0,10,100)
theta_vals = theta_func(t_vals)
plt.figure(figsize=(10, 5))
plt.plot(t_vals,theta_vals,linewidth=2)
plt.xlabel('t')
plt.ylabel(r'$\theta(t)$')
plt.title(r'$\theta(t)$ vs t')
plt.grid(True)
plt.show()