"""
Python Project
Subject: Mathematics
Focus: Solving a non-linear equation
"""

#Importing Libraries such as matplotlib, numpy and sympy
import matplotlib.pyplot as plt
import numpy as np
from sympy.solvers import solve
from sympy import N,re,im,diff, symbols,Abs
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.str import StrPrinter

#Function for the equation:
def constructed_exp(kappa, m, rho, sigma, C, Phi):
    expr_str = f"({kappa}**2)/(2*({m}**2 + {rho}*{kappa}))*{Phi}**3 + (3*{sigma}*{kappa})/(2*({m}**2 + {rho}*{kappa}))*{Phi}**2 + ({sigma}**2)/(2*({m}**2 + {rho}*{kappa}))*{Phi} + ({C}*{kappa})/(2*({m}**2 + {rho}*{kappa}))"
    return expr_str

# Defining the Axis of the plot
fig,axis = plt.subplots(1,2,figsize=(12,5))

#Taking values of parameters from the user:
try:
    kappa = float(input("Enter the value for κ (kappa): "))
    m = float(input("Enter the value for m: "))
    rho = float(input("Enter the value for ρ (rho): "))
    sigma = float(input("Enter the value for σ (sigma): "))
    C = float(input("Enter the value for C: "))
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()

Phi = symbols('Phi')

#Evaluating and simplifying the function values

expr_str = constructed_exp(kappa, m, rho, sigma, C, Phi)
print(f"Constructed expression string: {expr_str}")

expr = parse_expr(expr_str)
print(f"Parsed expression: {expr}")

formatted_expr = str(expr).replace("Phi","Φ")
print(formatted_expr)

#Finding the Roots of the function and displaying output

try:
    res = solve(expr, Phi)
    print(f"Solutions: {res}")
    stability_results = []
    stability_results_abs = []

    for i, r in enumerate(res):
        value = N(re(r),6)
        output = f'Root {i+1} is {value}'
        print(output)

        # Evaluate the derivative at the root
        derivative = diff(expr,Phi)  # Is line me derivative compute kre ge w.r.t PHI(Φ)...Function(function asal me expression{f(x)} hai) ka beahviour chck kr ske
        stability = derivative.subs(Phi,r)# Ye opr wale variable(derivative) me PHI ko replace krde ga root ke sath or phr derivative calculte kre ga
        # Baki neche ez hai
        # Check if the stability value is a real number
        stability_status_re = "Unstable" if re(stability) > 0 else "Stable"
        stability_results.append((value, stability_status_re))

        print(f"Stability at Root {i + 1}: {stability_status_re} (Derivative: {stability})")

        # Checking stability value by evaluating absolute
        absolute_res = "Stable" if Abs(N(Abs(stability),6)) <= 1 else "Unstable"
        stability_results_abs.append((r,absolute_res))

    # ALL Stability Results
    print("\n Stability Analysis by taking real part of Eigenvalues: ")
    for root, status in stability_results:
        print(f"Root: Φ = {root}, Status: {status}")

    print("\n Stability Analysis by taking absolute value of Eigenvalues: ")
    for root, status in stability_results_abs:
        print(f"Root: Φ = {root}, Status: {status}")
except Exception as e:
    print(f"An error occurred: {e}")

#Plotting The function on a Graph
x = np.linspace(-10,10,10000)
y = []
for n in x:
    y_val = (kappa**2)/(2*(m**2 + rho*kappa))*n**3 + (3*sigma*kappa)/(2*(m**2 + rho*kappa))*n**2 + (sigma**2)/(2*(m**2 + rho*kappa))*n + (C*kappa)/(2*(m**2 + rho*kappa))
    y.append(y_val)

axis[0].plot(x,y,c="red")
axis[0].axvline(x=0,c="black",linewidth=0.5)
axis[0].axhline(y=0,c="black",linewidth=0.5)
axis[0].set_xlabel("Φ")
axis[0].set_ylabel("f(Φ)")
axis[0].set_title("Graph of f(Φ)")
axis[0].grid()

# Sensitive Analysis
def func(kappa,m,rho,sigma,C,n):
    return (kappa**2)/(2*(m**2 + rho*kappa))*n**3 + (3*sigma*kappa)/(2*(m**2 + rho*kappa))*n**2 + (sigma**2)/(2*(m**2 + rho*kappa))*n + (C*kappa)/(2*(m**2 + rho*kappa))

size = 10000
kappa_samples = np.random.uniform(0.1,1,size)
m_samples = np.random.uniform(0.1,1,size)
rho_samples = np.random.uniform(0.1,1,size)
C_samples = np.random.uniform(0.1,1,size)
sigma_samples = np.random.uniform(0.1,1,size)
n_samples = np.random.uniform(0,100,size)

outputs = func(kappa_samples,m_samples,rho_samples,sigma_samples,C_samples,n_samples)
output_variance = np.var(outputs)

print(f"\n Variance of output: {output_variance}")

kappa_contribution = np.var(func(kappa_samples,m_samples.mean(),rho_samples.mean(),sigma_samples.mean(),C_samples.mean(),n_samples)) / output_variance
print(f"Sensitivity of kappa: {kappa_contribution}")

m_contribution = np.var(func(kappa_samples.mean(),m_samples,rho_samples.mean(),sigma_samples.mean(),C_samples.mean(),n_samples)) / output_variance
print(f"Sensitivity of m: {m_contribution}")

rho_contribution = np.var(func(kappa_samples.mean(),m_samples.mean(),rho_samples,sigma_samples.mean(),C_samples.mean(),n_samples)) / output_variance
print(f"Sensitivity of rho: {rho_contribution}")

C_contribution = np.var(func(kappa_samples.mean(),m_samples.mean(),rho_samples.mean(),sigma_samples.mean(),C_samples,n_samples)) / output_variance
print(f"Sensitivity of C: {C_contribution}")

sigma_contribution = np.var(func(kappa_samples.mean(),m_samples.mean(),rho_samples.mean(),sigma_samples,C_samples.mean(),n_samples)) / output_variance
print(f"Sensitivity of sigma: {sigma_contribution}")

# Plotting the Analysis
k = {"κ":kappa_contribution,
     "m":m_contribution,
     "ρ":rho_contribution,
     "C":C_contribution,
     "σ":sigma_contribution}

x_values = k.keys()
y_values = k.values()
axis[1].bar(x_values,y_values,0.2,color="red")
axis[1].set_xlabel("Parameters")
axis[1].set_ylabel("Sensitivity")
axis[1].set_title("Sensitivity Analysis")

# Plotting the Stability
# Marking the roots on the plot
for root, status in stability_results:
    axis[0].plot(root, 0, marker='o', markersize=8, label=f'{status} At: Φ={root:.3f}', color='green' if status == "Stable" else 'blue')

# Formatting
fig.canvas.manager.set_window_title('Equation Analysis')
fig.text(0.5,0.95,f"Equation f(Φ) = {formatted_expr}",fontsize='12',fontweight='400',ha='center',va='top')
fig.text(0.5,0.88,f"Roots (Equilibrium Points)",fontsize='14',fontweight='600',ha='center',va='top')
fig.text(0.5,0.82,f"Φ={N(res[0],6)}",fontsize='14',fontweight='400',ha='center',va='top')
fig.text(0.5,0.76,f"Φ={N(res[1],6)}",fontsize='14',fontweight='400',ha='center',va='top')
fig.text(0.5,0.70,f"Φ={N(res[2],6)}",fontsize='14',fontweight='400',ha='center',va='top')
fig.text(0.5,1,f"Parameters Passed: κ: {kappa}, m: {m}, ρ: {rho}, σ: {sigma}, C: {C}",fontsize='16',fontweight='400',ha='center',va='top')
fig.subplots_adjust(wspace=1,left=0.06,right=0.97)
axis[0].legend(title='For real parts')

# Showing
axis[0].grid()
plt.show()
