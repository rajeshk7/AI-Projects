import numpy as np
import random
import math
import sys
import random
import csv
import copy
import matplotlib.pyplot as plt

y_value=[]
var_g = 0

def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):

        hypothesis = np.dot(x, theta)
        loss = hypothesis - y

        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)

        cost = np.sum(loss ** 2) / (2 * m)

        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        #print(i,hypothesis,y,loss,alpha*gradient)
        # update
        theta = theta - alpha * gradient
    #print(hypothesis)
    return theta

#binary_v is binary vectory where 0 mean feature should be avoided and 1 means the coressponding
#feature should be added and total number of set bit denotes the number of inpute deature in consideration.
#input vector is the actual input and binary vector is sort of a mask that filters out redundant feature
def eval(input_vector,binary_v,var_g):
    # total depedent variable should be number of set bit in binary_v
    # x,y,m should be calculated from input_vector
    #input_vector will read data from .csv file

    n = input_vector.shape[1]   #Total features
    t = input_vector.shape[0]   #Total training size
    m = int(.8 * t)
    t = t - m


    var = 0

    for i in range(n-1):
        if binary_v[0][i] == True:
            var = var + 1

    x = np.zeros((m,var+1))
    y = np.zeros((m,1))

    for i in range(m):
        x[i][0] = 1.0;
        k = 1
        for j in range(n-1):
            if binary_v[0][j] == True:
                x[i][k] = float(input_vector[i][j]);
                k = k + 1
        y[i][0] = float(input_vector[i][n-1]);


   # x = np.matrix(x);
   # y = np.matrix(y);

    numIterations = 1000 + var_g*10;
    alpha = 0.0005
    theta = np.zeros(shape=(var+1,1))    #A matrix of n+1 zeros


    theta = gradientDescent(x, y, theta, alpha, m, numIterations)

   # print ("theta",theta)
    t_matrix = np.zeros((t,var+1))

    finall = np.zeros((t,1))
    fine = np.zeros((t,1))

    for i in range(t):
        t_matrix[i][0] = 1
        k = 1
        for j in range(n-1):
            if binary_v[0][j] == True:
                t_matrix[i][k] = float(input_vector[m+i][j]);
        finall[i] = float(input_vector[m+i][n-1]);


    t_matrix = np.matrix(t_matrix);



    final = t_matrix * theta;

    var = 0;

    for i in range(t):
        fine[i] = finall[i] - final.item(i,0);
        #print (finall[i]);
        fine[i] = fine[i]/finall[i];
        var = var + abs(fine[i]);

    var = var/t;
    var = 1 - abs(var);

    return var*100

#calculate the probality to move to neighbour n of current state c
def probabilty(delta_E, T):
    return  math.exp(delta_E/T)

def search_SA(input_v,sol):  # sol is a boolean array and 0 represents absence of input feature 1 repesent presence of the same
    n = sol.shape[1]
    vat = 0
    vati = 0
    acc = 1.0
    pv_acc = 0.5
    next_sol=copy.copy(sol)
    while (True):
        e = random.randint(0,n-1) #Starting from a random point
        if sol[0][e]==True:
            next_sol[0][e]=False
        else:
            next_sol[0][e]=True
        e2=eval(input_v,next_sol,vat)
        e1=eval(input_v,sol,vat)
        diff =  e2- e1
        vat = vati*10;
        if diff > 0:
            sol=copy.copy(next_sol)
        if vati > 60:
            if diff > 0:
                sol=copy.copy(next_sol)
            else:
                break
        vati +=1
        acc=eval(input_v,sol,vat)
        
        if(acc > pv_acc):
            y_value.append(acc)
            pv_acc = acc
            #print (acc)
    return sol


def nomalization(input_v):
    input_v=(input_v-input_v.mean(0))/np.std(input_v,axis=0)
    return input_v

reader = csv.reader(open("test.csv", "r"), delimiter=",")
x=list(reader)
input_v = np.array(x).astype("float")
input_v[:,0:input_v.shape[1]-1]=nomalization(input_v[:,0:input_v.shape[1]-1])
sol=np.full((1, (input_v.shape[1] -1 )), True, dtype=bool)
sol = search_SA(input_v,sol)
plt.plot(y_value)
plt.title('Iteration vs Accuracy')
plt.ylabel('Accuracy in percentage')
plt.xlabel('number of iteration')
plt.show()
