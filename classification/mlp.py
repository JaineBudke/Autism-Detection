from random import seed
import random
import math


####################################
#                                  #
#  Backpropagation Neural Network  #
#                                  #
####################################
# Code based on: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
class Mlp:


    layers = []
    network = []

    # layers: list of dicts
    # format: [{'neurons':number_of_neurons, 'act_func':activation_function}, ...]
    # len(layers)-2: number of hidden layers
    # layers[0]: input layer
    # layers[-1]: output layer
    # Example: [{'neurons': 10, 'act_func': ''}, {'neurons': 2, 'act_func': 'relu'}, {'neurons': 2, 'act_func': 'sigmoid'}]
    def __init__(self, layers):
        self.layers = layers
        self.network = []


    # Initialize the network with uniform distribution
    def initialize_network(self):

        # for each layer on network (except input layer, initialize on 1: first hidden layer)
        for index_layer in range(1,len(self.layers)):

            current_layer = self.layers[index_layer]
            prev_layer = self.layers[index_layer-1]

            # each neuron on current layer has a dict with n weights that corresponds on prev layer neurons
            # add type of activation function 
            new_layer = [{'weights':[random.uniform(0,1) for i in range(prev_layer['neurons'] + 1)], 
                          'act_func':current_layer['act_func']} for i in range(current_layer['neurons'])]
            
            self.network.append(new_layer)
        

    # Calculate neuron activation for an input
    def activate(self, weights, inputs):

        # initialize with bias
        activation = weights[-1] 

        # b + sum ( w * input )
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]

        return activation


    # Apply activation function on sum
    def activationFunction(self, activation, act_func):
        if(act_func == "sigmoid"):
            return 1.0 / (1.0 + math.exp(-activation))
        elif(act_func == 'tanh'):
            return (1.0 - math.exp(-activation)) / (1.0 + math.exp(-activation))
        elif(act_func == 'relu'):
            return max(0, activation)


    # Forward propagate input to output
    def forward_propagate(self, row):

        # the first input is the current row in the dataset
        inputs = row

        for layer in self.network:
            outputs = []
            for neuron in layer:
                
                # sum ( w * input ) + b
                activation = self.activate(neuron['weights'], inputs) 
                # apply activation function 
                neuron['output'] = self.activationFunction(activation, neuron['act_func'])
                outputs.append( neuron['output'] )
                
            # the outputs of current layer will be the inputs of next layer
            inputs = outputs

        return inputs


    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output, act_func):

        if(act_func == "sigmoid"):
            return output * (1.0 - output)
        elif(act_func == 'tanh'):
            return (1.0/2.0)*(1.0 - (output ** 2) )
        elif(act_func == 'relu'):
            if(output >= 0):
                return 1.0
            else:
                return 0.0


    # Backpropagate error gradient (delta) and store in neurons
    def backward_propagate_error(self, expected):

        # back through the network, from the output layer to the input layer 
        for layer_index in reversed(range(len(self.network))):
            
            # get current layer
            layer = self.network[layer_index]

            # initialize error signal list
            errors = list()
            
            # check if it's the output layer 
            if layer_index == len(self.network)-1:

                # for each neuron in the output layer
                for j in range(len(layer)):
                    # get current neuron
                    neuron = layer[j]
                    # add error signal on list
                    errors.append( expected[j] - neuron['output'] )

            # if not the output layer 
            else:
                
                # for each neuron in the current layer
                for neuron_index in range(len(layer)):
                    
                    error = 0.0

                    # for each neuron in the next layer
                    for neuron in self.network[layer_index + 1]:

                        # (w * delta)
                        # w: edge between next and current neuron; delta: error gradient we want to propagate 
                        error += (neuron['weights'][neuron_index] * neuron['delta'])
                    
                    # add error signal on list
                    errors.append(error)
                    

            # for each neuron in the current layer, calculate delta with error signal list
            for j in range(len(layer)):

                # get current neuron
                neuron = layer[j]

                # delta (error gradient) = error * act function derivative
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'], neuron['act_func'])


    # Update network weights with error
    def update_weights(self, row, l_rate):

        # for each layer in the network
        for i in range(len(self.network)):

            # initialize with row of dataset (except the label - last position) 
            out = row[:-1]
            
            # check if it's not the input layer
            if i != 0:
                # get output list from prev layer (will be the input from current layer)
                out = [neuron['output'] for neuron in self.network[i - 1]]

            # for each neuron on current layer
            for neuron in self.network[i]:

                # for each output from the prev layer (or row of dataset) 
                for j in range(len(out)):

                    # update the weight 
                    neuron['weights'][j] += l_rate * neuron['delta'] * out[j]
                
                # update the bias
                neuron['weights'][-1] += l_rate * neuron['delta']


    # train network
    def fit(self, n_epoch, dataset, l_rate):

        # initialize weights
        self.initialize_network()

        # stop criteria: number of epochs
        for epoch in range(n_epoch):
            
            sum_error = 0

            # for each row on dataset
            for row in dataset:

                # propagate row data from input layer to output layer 
                outputs = self.forward_propagate(row)
                
                # creates a vector of 0s with the size of the output layer, to represent the expected output 
                expected = [0 for i in range(self.layers[-1]['neurons'])]

                # set to 1 the correct output position (each position of the vector represents a class, 0 to 1st, 1 to 2nd etc) 
                # example: [0,0,1] means that correct class is the 3nd 
                expected[int(row[-1])] = 1
                
                # calculate error signal to print
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                
                # propagates error with gradient descent
                self.backward_propagate_error(expected)

                # update network weights
                self.update_weights(row, l_rate)

            print('>epoch=%d, error=%.3f' % (epoch, sum_error))


    # Make a prediction with the MLP
    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))


    # Predict dataset 
    def predictAll(self, X_test):
        predictions = list()
        for row in X_test:
            prediction = self.forward_propagate(row)
            predictions.append(prediction.index(max(prediction)))
        return(predictions)
