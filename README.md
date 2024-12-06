# Equation Analysis Project

This project solves and analyzes non-linear equations using Python libraries like Sympy, Matplotlib, and Numpy. It calculates the roots of the equation, performs stability analysis, and evaluates the sensitivity of the parameters.

## Features
- Dynamically constructs and solves equations.
- Finds roots of the equation symbolically.
- Plots the function behavior on a graph.
- Conducts stability and sensitivity analysis.

## How to Run

1. **Install the required libraries:**

   Before running the script, you need to install some Python libraries. These libraries help us with solving equations, plotting graphs, and doing mathematical calculations. 

   To install the libraries, open your terminal (command prompt) and type the following command: pip install sympy numpy matplotlib
   
Press Enter after typing this command. This will automatically download and install the required libraries for the project.

2. **Run the script:**

After the libraries are installed, you can run the Python script to solve the equations. In the terminal (command prompt), type this command to start the program: python math_project.py

This will run the Python script, and the program will start working. It will ask you to enter some values (parameters) for the equation.

3. **Input Parameters:**

When you run the script, the program will ask you to enter values for the following parameters:

- κ (kappa)
- m
- ρ (rho)
- σ (sigma)
- C

You will type these values into the terminal when prompted. Make sure to enter numbers for each of them.

4. **Expected Output:**

After entering the values for the parameters, the program will show the following results:

- **Roots of the equation:**
  The script will calculate the roots (solutions) of the equation. These are the points where the function equals zero. The program will print out these roots in the terminal, for example:
  ```
  Roots: Φ = 2.45, Φ = -1.12, Φ = 0.87
  ```

- **Graph of the equation:**
  The program will plot a graph of the equation using **Matplotlib**. You will see a graph that shows the behavior of the equation based on the values you entered. The graph will display:
  - The function's curve.
  - The roots where the curve crosses the x-axis.

- **Stability Analysis:**
  The program will perform a **stability analysis** on the roots. It will print whether each root is **stable** or **unstable**. For example, you might see:
  ```
  Stability at Root 1: Stable
  Stability at Root 2: Unstable
  Stability at Root 3: Stable
  ```

- **Sensitivity Analysis:**
  The program will perform a **sensitivity analysis** to show how each parameter (κ, m, ρ, σ, C) affects the output. It will generate a **bar chart** that shows the sensitivity of each parameter:
  - The x-axis of the bar chart represents the parameters.
  - The y-axis represents the sensitivity (how much that parameter affects the results).

  Example of output for sensitivity analysis:
  ```
  Sensitivity of κ (kappa): 0.75
  Sensitivity of m: 0.20
  Sensitivity of ρ (rho): 0.15
  Sensitivity of σ (sigma): 0.10
  Sensitivity of C: 0.05
  ```





   
