import numpy as np
import random
from matplotlib import pyplot as plt
import math
#variables
A = 0.3 #learning rate
total_error=0
topology = [2,10,1] # size of the network
layers =[]
weights =[]

class Neuron(object):
    
    def __init__(self):
        self.value = 0
        self.deriv_value = 0
        self.error = 0      
    def activate(self):
        self.value = math.e**(self.value)/(math.e**(self.value)+1)
    def deriv(self):   
        self.deriv_value = self.value * (1 -self.value)
        return self.deriv_value
        
class Layer(object):
    
    def __init__(self):
        self.neurons =[]
    def values(self):
        vals =[]
        for n in self.neurons:
            vals.append(n.value)
        return np.asmatrix(vals) 
    def errors(self):
        err =[]
        for n in self.neurons:
            err.append(n.error)
        return np.asmatrix(err)
#methods
def feed_layer(layer_index, values): #feed layer with given values
    for i in range(len(layers[layer_index].neurons)):
        if layer_index>0: #is it 1st layer?
            layers[layer_index].neurons[i].value = values[i]
            layers[layer_index].neurons[i].activate() #activate values
        else:
            layers[layer_index].neurons[i].value = values[i]
        
def feed_forward(): #calculate values every layer
    for i in range(1,len(layers)):
        r = np.dot(layers[i-1].values(),weights[i-1])
        r = np.asarray(r)
        feed_layer(i,r[0]) #[0]because of datatype
        
def set_error(target):
    i = 0
    _total_error =0
    for n in layers[len(layers)-1].neurons: #last layer errors
        n.error = target[i] - n.value
        _total_error += n.error
        i+=1
    for k in range(len(topology)-1,0,-1): #set errors backwards
        r = np.dot(weights[k-1],layers[k].errors())
        r = np.asarray(r)
        for j in range(len(layers[k-1].neurons)):
            layers[k-1].neurons[j].error = r[j]
    return _total_error

#update weights
def update_weights():
    k = 1
    for w in weights: 
        for i in range(w.shape[0]): # in rows
            for j in range(w.shape[1]): # in columns
                w[i][j] += A*layers[k].neurons[j].error*layers[k].neurons[j].deriv()*layers[k-1].neurons[i].value
        k+=1
        
#initialize network
for layer in topology: #create layers and fill with default neurons
    l = Layer()
    for i in range(layer):
        n = Neuron()
        l.neurons.append(n)
    layers.append(l)
    
out = layers[len(layers)-1].neurons[0] #output layer

for i in range(len(topology)-1): #generate weight matrices
    weights.append(np.random.rand(topology[i],topology[i+1]))
    
#train - XOR gate
for i in range(30000):
    x = random.randint(0,1)
    y = random.randint(0,1)
    target = np.array([1])
    if x == y:
        target = np.array([0])
    feed_layer(0,(x,y))
    feed_forward()
    set_error(target)
    update_weights()
    #print(x,y, out.value)

#test
right = 0
zeros = 0
ones = 1
n = 1000
for i in range(n): 
    x = random.randint(0,1)
    y = random.randint(0,1)
    target = np.array([1])
    if x == y:
        target = np.array([0])
    
    feed_layer(0,(x,y))
    feed_forward()
    result = 1
    if out.value<0.1:
        result = 0
        zeros += 1
    else:
        ones += 1
    if target == result:
        right +=1

print("zeros on out:",zeros, "ones on out:",ones)
print("percentage of positive answers:(not accuracy)",100*right/n)

#draw graph of accuracy
right = 0
zeros = 0
ones = 1
arr = np.zeros((101,101))
for j in range(101):
    for k in range(101):
        feed_layer(0,(j/100,k/100))
        feed_forward()
        arr[j,k] = out.value
arr *=255
plt.imshow(arr)
plt.show()