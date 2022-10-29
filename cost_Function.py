# printing in python
print("This is  code cell")


# printing variables and text 

variable = "right in the strings!"
print(f"f strings allow you to embed variables {variable}")

# importing and using numpy library  => a popular library for scientific computing
import numpy as np

# importing and using Matplotlib library  => a popular library for plotting data
import matplotlib.pyplot as plt

# using style package of matplotlib to select a style => style selected: deeplearning.mplstyle
plt.style.use('./deeplearning.mplstyle')

################################################################ Arrays for training set #####################################################################

# creating arrays 

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# printing values in arrays 
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# returning a python tuple with entry for each dimension 
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")

# .shape[0] returns length of the array 
m = x_train.shape[0]
# equivelent way =========>>>>>>   m = len(x_train)

# printing m
print(f"Number of training examples is: {m}")

#defining a variable i
i = 1

# using variable i to assign values in array to variables 
# note: their name itself contain i!!!! that is OK !!!!! 
x_i = x_train[i]
y_i = y_train[i]

# printing the new variables 
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

####################################################################### plotting #################################################################################


# Plotting data points
plt.scatter(x_train, y_train, marker='x', c='r')

# Setting the title / x-axis label / y-axis label
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')

# showing the plot 
plt.show()


################################################################ working on the model funciton ##################################################################

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

#############################  defining funciton to calculate the mathematical formula for the linear model  

def compute_model_output(x, w, b):
    # making a python docstring to document the function 
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    #getting value of m from function arguments
    m = x.shape[0]
    
    # defining variable f_wb to be a one-dimensional numpy array with  m  entries 
    f_wb = np.zeros(m)
    
    # for loop
    for i in range(m):
        # filling f_wb array with the function's calculated values 
        f_wb[i] = w * x[i] + b
        
    return f_wb
  
#############################  using the linear model function I built  
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plotting linear funciton ( the prediction) 
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plottinh data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title / x-axis label / y-axis label
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')

# showing the legend and showing the graph I plotted 
plt.legend()
plt.show()


# using values of w and b of the model to predict value of y at a certain x
w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")
























