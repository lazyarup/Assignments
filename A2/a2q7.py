import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
#a
f=lambda y,x:10-8*x**2-2*y**2
x_lower = 0
x_upper = 1
y_lower = 0
y_upper = 2
total_temperature=sp.integrate.dblquad(f,x_lower,x_upper,y_lower,y_upper)[0]
print(f'The total temperature is: {total_temperature}\n')
area=(x_upper-x_lower)*(y_upper-y_lower)
avg_temp=total_temperature/area
print(f'the average temperature of the rectangular portion is: {avg_temp} degrees Celsius')

#b
import sympy as smp
t=smp.symbols('t',real=True)
x,y,z,r,f=smp.symbols('x y z r f',cls=smp.Function,real=True)
x=x(t)
y=y(t)
z=z(t)
r=smp.Matrix([x,y,z])
f=f(x,y,z)
integrand=f*smp.diff(r,t).norm()
integrand=integrand.subs([(f,x*y+z**3),(x,smp.cos(t)),(y,smp.sin(t)),(z,t)]).doit().simplify()
integrand_2=smp.lambdify([t],integrand)
line_integral=sp.integrate.quad(integrand_2,0,np.pi)[0]
print(f'the values of the line integralis:{line_integral}\n')

th=np.linspace(0,np.pi,100)
x1=np.cos(th)
y1=np.sin(th)
z1=th

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

pl1=ax.plot(x1,y1,z1)

ax.scatter(1,0,0,label='(1,0,0)')
ax.scatter(-1,0,np.pi,label='(-1,0,3.1416)')

plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
ax.set_title('Helix C and Line Integral')
ax.legend()

plt.show()


r,theta,z,h=smp.symbols('r theta z h' ,real=True)
rho=r**2
dV=r*smp.diff(r)*smp.diff(theta)*smp.diff(z)
mass_integral=smp.integrate(rho*dV,(r,0,r),(theta,0,2*smp.pi),(z,0,h)).doit().simplify()
print(f'the mass of the cylinder is: {mass_integral}\n')

#let
h=2
r=1
theta=np.linspace(0,2*np.pi,100)
x=r*np.cos(theta)
y=r*np.sin(theta)
Z1=np.linspace(0,h,100)

fig=plt.figure(figsize=(12,8))

ax=fig.add_subplot(111,projection='3d')

for z in Z1:
 ax.plot(x,y,z,color='cyan',alpha=0.5)

plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z') 

plt.show()

x,y,phi=smp.symbols('x y phi',real=True)
F1,F2,F=smp.symbols('F1 F2 F',cls=smp.Function,real=True)
F1=F1(x,y)
F2=F2(x,y)
F=smp.Matrix([F1,F2])
F1=smp.exp(y)
F2=x*smp.exp(y)
df1=smp.diff(F1,y)
df2=smp.diff(F2,x)

if df1==df2:
    print('the force field F is conservative \n')

PHI_x=F1
PHI_y=F2 

phi=smp.integrate(PHI_x,x)
C_y=phi+smp.Symbol('C')
C=smp.diff(C_y,y)-PHI_y
phi=phi+C
print(f'the potential function is: {phi.simplify()}\n')

phi_A=phi.subs([(x,1),(y,0)])
phi_B=phi.subs([(x,-1),(y,0)])
Work_done=phi_B-phi_A
print(f'work done: {Work_done}\n')

fig=plt.figure(figsize=(10,8))

x=np.linspace(-2,2,20)
y=np.linspace(-2,2,20)
X,Y=np.meshgrid(x,y)
U=np.exp(Y)
V=X*np.exp(Y)
plt.quiver(X,Y,U,V,color='blue',alpha=0.5)

theta=np.linspace(0,np.pi,100)
x1=np.cos(theta)
x2=np.sin(theta)
plt.plot(x1,x2,color='green')
plt.scatter(1,0,color='red',label='Start(1,0)')
plt.scatter(-1,0,color='purple',label='End(-1,0)')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Force Field and Semicircular Path')
plt.legend()
plt.grid()
plt.show()