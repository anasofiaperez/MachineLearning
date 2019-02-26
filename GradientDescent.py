# CREATED BY ANA PEREZ RODRIGUEZ
# MACHINE LEARNING ASSIGNMENT 1
# GRADIENT DESCENT ALGORITHM FOR MULTI-VARIABLE LINEAR REGRESSION

# Part 1: Download the dataset and partition it randomly into train and test set using a 70/30 split.
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt
from tabulate import tabulate

# Set working directory
os.chdir('C:\\Users\\anaso\\Desktop\\Machine Learning\\Assignment1\\Data')

# Listing file names we will read: Features Variant 1 plus all Test Cases
file_names=['Features_Variant_1.csv']
for i in range(1,11):
    file_name= 'Test_Case_' + str(i) +'.csv'
    file_names.append(file_name)

# Storing data in a data frame
list_data=[]
for filename in file_names:
    data = pd.read_csv(filename, header=None)
    list_data.append(data)
data=pd.concat(list_data, ignore_index=True)            #data.reset_index(drop=True)

#Creating training and testing sets
#np.random.seed(10)
train, test = train_test_split(data, test_size=0.3,random_state=1)  # print(train) #29364 is approx 41949*.7
#print(train.head(5))
train.reset_index(drop=True,inplace=True)
test.reset_index(drop=True,inplace=True)

train2=train.copy()
test2=test.copy()

# Normalize Data
train2 = (train2 - train2.mean())/train2.std()
test2=(test2 - test2.mean())/test2.std()

# Part 2: Design a linear regression model to model the number of comments a post will receive in next H
# hours. Include your regression model equation in the report.
var_list=[1,2,3]

def mul_lin(var_list=[1],alpha=.01, threshold=.1,max_iter=100, train2=train2):
    #seed(1)
    m = train2.shape[0]
    x0 = np.ones(m)

    num_vars = len(var_list)
    X = [x0]

    for var in var_list:
        X.append(np.array(train2[var]))
    X = np.array(X)
    X = X.T

    # We will start with small random numbers for betas
    beta = []
    random.seed(1)
    for var in range(num_vars + 1):
        beta.append(random.random())
    B = np.array(beta)
    y_pred = X.dot(B)

    y = np.array(train2[53])
    cost = (1 / 2 / m) * sum(diff ** 2 for diff in (y_pred - y))

    iter = 0
    converged = False
    # Used 7 as a place value, no particular meaning
    gradient = [7] * len(B)
    temp_beta = [7] * len(B)

    while converged==False:
        # B0 gradient calculation
        gradient[0] = (1 / m) * sum(y_pred-y)

        # Bi gradient calculation
        for i in range(1,len(B)):
            gradient[i]= (1 / m) * sum((y_pred-y)*X.T[i])

        #Make temp betas
        for i in range(len(B)):
            temp_beta[i]=B[i]-alpha*gradient[i]

        #Update Betas
        for i in range(len(B)):
            B[i]=temp_beta[i]

        #Predict y by multiplying xs by betas
        y_pred = X.dot(B)
        new_cost= (1 / 2 / m) * sum(diff ** 2 for diff in (y - y_pred))

        if abs(cost-new_cost) <= threshold:
            converged = True
            break


        # update error and number of iterations
        cost = new_cost
        iter += 1

        if iter == max_iter:
            #print('Max interactions exceeded!')
            converged = True

    return B, new_cost, iter



# Testing our model in the test data set
def test_error(test2=test2, var_list=[1], beta_out=np.array([1])):

    m = test2.shape[0]
    x0 = np.ones(m)
    X_test = [x0]

    for var in var_list:
        X_test.append(np.array(test2[var]))

    X_test = np.array(X_test)
    X_test =X_test.T
    y_pred_test = X_test.dot(beta_out)
    y_test= np.array(test2[53])
    cost_test = (1 / 2 / m) * sum(diff ** 2 for diff in (y_pred_test - y_test))

    return cost_test



# Part 3: Implement the gradient descent algorithm with batch update rule. Use the same cost function as
# in the class (sum of squared error). Report your initial parameter values.


# 1. Experiment with various values of learning rate alpha and report on your findings as how the error
# varies for train and test sets with varying alpha. Plot the results. Report your best alpha and why you
# picked it.
var_exp1=[1,2,3,5,7,8,9,10,11,31,38]
alpha_list=[0.001,0.01,0.05,0.1,0.2,0.3,0.4]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for a in alpha_list:
    new_beta, new_cost,iter_n=mul_lin(var_exp1, a, threshold=.01, max_iter=100, train2=train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, var_exp1, new_beta)
    list_test_error.append(tst_err)


print(list_cost)
print(list_test_error)
print(list_iter)



# Convert lists into arrays to plot
np.asarray(alpha_list)
np.asarray(list_test_error)
np.asarray(list_cost)

# Plot results
plt.title("Alpha value vs. Test Error")
plt.xlabel("Alpha value")
plt.ylabel("Test Error")
plt.plot(alpha_list,list_test_error)
plt.show()

plt.title("Alpha value vs. Train Error")
plt.xlabel("Alpha value")
plt.ylabel("Train Error")
plt.plot(alpha_list,list_cost)
plt.show()


var_exp1=[1,2,3,5,7,8,9,10,11,31,38]
alpha_list=[0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for a in alpha_list:
    new_beta, new_cost,iter_n=mul_lin(var_exp1, a, threshold=.01, max_iter=100, train2=train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, var_exp1, new_beta)
    list_test_error.append(tst_err)


print(list_cost)
print(list_test_error)
print(list_iter)



# Convert lists into arrays to plot
np.asarray(alpha_list)
np.asarray(list_test_error)
np.asarray(list_cost)

# Plot results
plt.title("Alpha value vs. Test Error")
plt.xlabel("Alpha value")
plt.ylabel("Test Error")
plt.plot(alpha_list,list_test_error)
plt.show()

plt.title("Alpha value vs. Train Error")
plt.xlabel("Alpha value")
plt.ylabel("Train Error")
plt.plot(alpha_list,list_cost)
plt.show()

# 2. Experiment with various thresholds for convergence. Plot error results for train and test sets as
# a function of threshold and describe how varying the threshold affects error. Pick your best
# threshold and plot train and test error (in one figure) as a function of number of gradient
# descent iterations.

var_exp1=[1,2,3,5,7,8,9,10,11,31,38]
alpha=0.4
list_thresh=[0.005,0.001,0.0005,0.0001,0.00005,0.00001]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for t in list_thresh:
    new_beta, new_cost, iter_n =mul_lin(var_exp1, .04, t, 100, train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, var_exp1, new_beta)
    list_test_error.append(tst_err)


#print(list_cost)
#print(list_test_error)
print(list_iter)




# Convert lists into arrays to plot
np.asarray(list_thresh)
np.asarray(list_test_error)
np.array(list_cost)

# Plot results
plt.title("Threshold vs. Test Error")
plt.xlabel("Threshold value")
plt.ylabel("Test Error")
plt.plot(list_thresh,list_test_error)
plt.show()

plt.title("Threshold vs. Train Error")
plt.xlabel("Threshold")
plt.ylabel("Train Error")
plt.plot(list_thresh,list_cost)
plt.show()



#Train and test error (in one figure) as a function of number of gradient descent iterations
alpha=0.4
list_iter2=[10,50,100,110,125,200,300]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for it in list_iter2:
    new_beta, new_cost, iter_n =mul_lin(var_exp1, .04, 0.00005, it, train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, var_exp1, new_beta)
    list_test_error.append(tst_err)


#print(list_cost)
#print(list_test_error)
print(list_iter)


# Convert lists into arrays to plot
np.asarray(list_iter2)
np.asarray(list_test_error)
np.array(list_cost)

#Plot results
plt.title("Number of Iterations vs. Test Error")
plt.xlabel("Number of Iterations")
plt.ylabel("Test Error")
plt.plot(list_iter2,list_test_error)
plt.show()

plt.title("Number of Iterations vs. Train Error")
plt.xlabel("Number of Iterations")
plt.ylabel("Train")
plt.plot(list_iter2,list_cost)
plt.show()

#Final linear model for random variables
new_beta, new_cost, iter_n =mul_lin(var_exp1, .4, 0.00005, 200, train2)
tst_err=test_error(test2, var_exp1, new_beta)
print("Betas: ",new_beta)
print("Test error: ", tst_err)
print("Train cost: ", new_cost)

# 3. Pick five features randomly and retrain your model only on these five features. Compare train
# and test error results for the case of using your original set of features (greater than 10) and five
# random features. Report which five features did you select randomly.
rand_vars = []
random.seed(3)
for var in range(5):
    rand_vars.append(random.randint(1,53))
print("Random variables: ", rand_vars)

alpha_list=[0.001,0.01,0.05,0.1,0.2,0.3,0.4, 0.5,0.6,0.65,0.7,0.8,1,1.2,1.3,1.4,1.5]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for a in alpha_list:
    new_beta, new_cost,iter_n=mul_lin(rand_vars, a, threshold=.01, max_iter=100, train2=train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, rand_vars, new_beta)
    list_test_error.append(tst_err)


print(list_cost)
print(list_test_error)
print(list_iter)

##build table

# Convert lists into arrays to plot
np.asarray(alpha_list)
np.asarray(list_test_error)
np.asarray(list_cost)

# Plot results
plt.title("Alpha value vs. Test Error")
plt.xlabel("Alpha value")
plt.ylabel("Test Error")
plt.plot(alpha_list,list_test_error)
plt.show()

plt.title("Alpha value vs. Train Error")
plt.xlabel("Alpha value")
plt.ylabel("Train Error")
plt.plot(alpha_list,list_cost)
plt.show()



list_thresh=[0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000001,0.0000001]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for t in list_thresh:
    new_beta, new_cost, iter_n =mul_lin(rand_vars, 0.9, t, 500, train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, rand_vars, new_beta)
    list_test_error.append(tst_err)


#print(list_cost)
print(list_test_error)
print(list_iter)



#
# # Convert lists into arrays to plot
np.asarray(list_thresh)
np.asarray(list_test_error)
np.array(list_cost)

# Plot results
plt.title("Threshold vs. Test Error")
plt.xlabel("Threshold value")
plt.ylabel("Test Error")
plt.plot(list_thresh,list_test_error)
plt.show()

plt.title("Threshold vs. Train Error")
plt.xlabel("Threshold")
plt.ylabel("Train Error")
plt.plot(list_thresh,list_cost)
plt.show()



# Train and test error (in one figure) as a function of number of gradient descent iterations

list_iter2=[4,5,6,7,8,9,10,11,12,13]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for it in list_iter2:
    new_beta, new_cost, iter_n =mul_lin(rand_vars, .9, 0.00005, it, train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, rand_vars, new_beta)
    list_test_error.append(tst_err)


#print(list_cost)
#print(list_test_error)
print(list_iter)


# Convert lists into arrays to plot
np.asarray(list_iter2)
np.asarray(list_test_error)
np.array(list_cost)

#Plot results
plt.title("Number of Iterations vs. Test Error")
plt.xlabel("Number of Iterations")
plt.ylabel("Test Error")
plt.plot(list_iter2,list_test_error)
plt.show()

plt.title("Number of Iterations vs. Train Error")
plt.xlabel("Number of Iterations")
plt.ylabel("Train")
plt.plot(list_iter2,list_cost)
plt.show()

#Final linear model for random variables
new_beta, new_cost, iter_n =mul_lin(rand_vars, .9, 0.00005, 15, train2)
tst_err=test_error(test2, rand_vars, new_beta)
print("Betas: ",new_beta)
print("Test error: ", tst_err)
print("Train cost: ", new_cost)

# 4. Now pick five features that you think are best suited to predict the output, and retrain your
# model using these five features. Compare to the case of using your original set of features and
# to random features case. Did your choice of features provide better results than picking random
# features? Why? Did your choice of features provide better results than using all features? Why?

chosen_vars=[1,2,3,31,38]
alpha_list=[0.001,0.01,0.05,0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for a in alpha_list:
    new_beta, new_cost,iter_n=mul_lin(chosen_vars, a, threshold=.01, max_iter=100, train2=train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, chosen_vars, new_beta)
    list_test_error.append(tst_err)


print(list_cost)
print(list_test_error)
print(list_iter)

# Convert lists into arrays to plot
np.asarray(alpha_list)
np.asarray(list_test_error)
np.asarray(list_cost)

# Plot results
plt.title("Alpha value vs. Test Error")
plt.xlabel("Alpha value")
plt.ylabel("Test Error")
plt.plot(alpha_list,list_test_error)
plt.show()

plt.title("Alpha value vs. Train Error")
plt.xlabel("Alpha value")
plt.ylabel("Train Error")
plt.plot(alpha_list,list_cost, list_test_error)
plt.show()



list_thresh=[0.001,0.0005,0.0002,0.0001,0.00005,0.00001,0.000001,0.0000001]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for t in list_thresh:
    new_beta, new_cost, iter_n =mul_lin(chosen_vars, 0.4, t, 500, train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, chosen_vars, new_beta)
    list_test_error.append(tst_err)


#print(list_cost)
print(list_test_error)
print(list_iter)




# Convert lists into arrays to plot
np.asarray(list_thresh)
np.asarray(list_test_error)
np.array(list_cost)

# Plot results
plt.title("Threshold vs. Test Error")
plt.xlabel("Threshold value")
plt.ylabel("Test Error")
plt.plot(list_thresh,list_test_error)
plt.show()

plt.title("Threshold vs. Train Error")
plt.xlabel("Threshold")
plt.ylabel("Train Error")
plt.plot(list_thresh,list_cost)
plt.show()



# Train and test error (in one figure) as a function of number of gradient descent iterations

list_iter2=[10,12,14,16,18,20,22]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for it in list_iter2:
    new_beta, new_cost, iter_n =mul_lin(chosen_vars, .4, 0.0001 ,it, train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, chosen_vars, new_beta)
    list_test_error.append(tst_err)


#print(list_cost)
#print(list_test_error)
print(list_iter)


# Convert lists into arrays to plot
np.asarray(list_iter2)
np.asarray(list_test_error)
np.array(list_cost)

#Plot results
plt.title("Number of Iterations vs. Test Error")
plt.xlabel("Number of Iterations")
plt.ylabel("Test Error")
plt.plot(list_iter2,list_test_error)
plt.show()

plt.title("Number of Iterations vs. Train Error")
plt.xlabel("Number of Iterations")
plt.ylabel("Train")
plt.plot(list_iter2,list_cost)
plt.show()

list_thresh=[0.001,0.0005,0.0002,0.0001,0.00005,0.00001,0.000001,0.0000001]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for t in list_thresh:
    new_beta, new_cost, iter_n =mul_lin(chosen_vars, 0.25, t, 500, train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, chosen_vars, new_beta)
    list_test_error.append(tst_err)


#print(list_cost)
print(list_test_error)
print(list_iter)




# Convert lists into arrays to plot
np.asarray(list_thresh)
np.asarray(list_test_error)
np.array(list_cost)

# Plot results
plt.title("Threshold vs. Test Error")
plt.xlabel("Threshold value")
plt.ylabel("Test Error")
plt.plot(list_thresh,list_test_error)
plt.show()

plt.title("Threshold vs. Train Error")
plt.xlabel("Threshold")
plt.ylabel("Train Error")
plt.plot(list_thresh,list_cost)
plt.show()



# Train and test error (in one figure) as a function of number of gradient descent iterations

list_iter2=[10,12,14,16,18,20,22]
list_beta=[]
list_cost=[]
list_iter=[]
list_test_error=[]

for it in list_iter2:
    new_beta, new_cost, iter_n =mul_lin(chosen_vars, .25, 0.0001 ,it, train2)
    list_beta.append(new_beta)
    list_cost.append(new_cost)
    list_iter.append(iter_n)

    tst_err=test_error(test2, chosen_vars, new_beta)
    list_test_error.append(tst_err)


#print(list_cost)
#print(list_test_error)
print(list_iter)


# Convert lists into arrays to plot
np.asarray(list_iter2)
np.asarray(list_test_error)
np.array(list_cost)

#Plot results
plt.title("Number of Iterations vs. Test Error")
plt.xlabel("Number of Iterations")
plt.ylabel("Test Error")
plt.plot(list_iter2,list_test_error)
plt.show()

plt.title("Number of Iterations vs. Train Error")
plt.xlabel("Number of Iterations")
plt.ylabel("Train")
plt.plot(list_iter2,list_cost)
plt.show()

# Final linear model for random variables
new_beta, new_cost, iter_n =mul_lin(chosen_vars, .25, 0.0001, 35, train2)
tst_err=test_error(test2, chosen_vars, new_beta)
print("Betas: ",new_beta)
print("Test error: ", tst_err)
print("Train cost: ", new_cost)
