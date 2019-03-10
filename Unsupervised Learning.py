'''
Joey Ma
Kohonen and K-means competitive learning ANN
'''

import csv
import numpy as np

#GLOBAL CONSTANTS
#1 for Kohonen, 2 for K-means
LEARNING_ALGORITHM = 2
#Learning rate
c = 0.5
Epoch = 15

#Reads csv
def read_csv():
    with open('dataset_noclass.csv','rt') as f:
        reader = csv.reader(f)
        data = list(reader)
    #Get rid of header
    inputs = data[1:]
    #Convert string to float
    inputs = [list(map(float,rec)) for rec in inputs]
    return inputs

#Prints the output to 2 decimal places without changing the output value
def To_decimal(weights):
    lis = weights
    for i in range(len(weights)):
        for k in range(len(weights[0])):
            lis[i][k] = round(weights[i][k],2)
    print(lis)

#Calculates the distance between the nodes and the weights
def EuclideanDistance(inputs, weights):
    d = (inputs[0] - weights[0]) ** 2 + (inputs[1] - weights[1]) ** 2 + (inputs[2] - weights[2]) ** 2
    #print(d)
    return d

#Gets the average of a cluster
def Cluster_Sum(Cluster):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    lis = []

    for i in Cluster:
        sum1 += i[0]
        sum2 += i[1]
        sum3 += i[2]
    lis.append(sum1 / len(Cluster))
    lis.append(sum2 / len(Cluster))
    lis.append(sum3 / len(Cluster))
    return lis

#Kohonen learning algorithm
def Kohonen_Learning(inputs, weights):
    for i in range(len(inputs)):
        lower = 99999
        for k in range(len(weights)):
            d = EuclideanDistance(inputs[i],weights[k])
            if d < lower:
                lower = d
                #Position of smallest distance in the weights list
                n = k
        for k in range(len(weights[n])):
            #Multiply and divide by 10 to avoid 0.1 + 0.2 == 0.30000000000000004
            weights[n][k] = weights[n][k] * 10 + c * (inputs[i][k] * 10 - weights[n][k] * 10)
            weights[n][k] = weights[n][k] / 10

#K-means algorithm
def K_Means(inputs, weights):
    Cluster1 = []
    Cluster2 = []

    for i in range(len(inputs)):
        lower = 99999
        for k in range(len(weights)):
            d = EuclideanDistance(inputs[i],weights[k])
            if d < lower:
                lower = d
                #Position of smallest distance in the weights list
                n = k
        if n == 0:
            Cluster1.append(inputs[i])
        else:
            Cluster2.append(inputs[i])

    print("Cluster 1:", len(Cluster1), "Cluster 2:", len(Cluster2))
    weights[0] = Cluster_Sum(Cluster1)
    weights[1] = Cluster_Sum(Cluster2)

def main():
    inputs = read_csv()

    #Make one of the weights 1,1,1
    weights = [[1,1,1]]
    lis = np.random.random_sample((1,3))
    weights.append(list(lis[0]))

    #Start epoch
    for epoch in range(Epoch):
        print("\nEpoch:", epoch + 1, "\n")
        if LEARNING_ALGORITHM == 1:
            Kohonen_Learning(inputs, weights)
        else:
            K_Means(inputs, weights)
        #Output
        To_decimal(weights)

main()
