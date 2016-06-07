__author__ = 'h_hack'
import numpy
import random
from activations import sigmoid, sigmoid_derivative

# defining a  neural network class and its attributes
class NeuralNetwork(object):
    def __init__(self, sizelist):
        # len(size) initializes the number of layers a neural network has
        self.layers = len(sizelist)
        # layerSize is the list of the sizes of each layer in neural network. It is passed as the argument.
        self.layersSize = sizelist[:]
        # weight matrix
        self.weights = [numpy.random.randn(nextdim, currentdim)+3 for currentdim, nextdim in zip(sizelist[:-1], sizelist[1:])]
        # bias matrix
        self.bias = [numpy.random.randn(currentlayer_size, 1) for currentlayer_size in sizelist[1:]]


    def feedingforward(self, A):
        for w, b in zip(self.weights, self.bias):
            A = numpy.dot(w, A) + b
            A = sigmoid(A)
        return A


    def backpropagate(self,z, train_out):
        z_list = []
        activations_list = []
        del_bias = [numpy.zeros(b.shape) for b in self.bias]
        del_weights = [numpy.zeros(w.shape) for w in self.weights]
        #print 'feed forwarding in back propagation.....'
        activations_list.append(z)
        for w, b in zip(self.weights, self.bias):
            z = numpy.dot(w, z) + b
            z_list.append(z)
            activation = sigmoid(z)
            activations_list.append(activation)
        #print 'feed forwarding done.....'

        #print 'calculating gradient and rho.....'
        #print activations_list[-1] - train_out
        gradient_c_wrt_a = activations_list[-1] - train_out
        #'sd:' , sigmoid_derivative(z_list[-1])
        rho = gradient_c_wrt_a * sigmoid_derivative(z_list[-1])
        #print 'calculated gradient and rho.....'

        #print activations_list
        #print 'rho:', rho
        del_bias[-1] = rho
        del_weights[-1] = numpy.dot(rho, activations_list[-2].transpose())

         #   print 'back propagating the error.....'
        for l in range(2, self.layers):
            k = z_list[-l]
            sd = sigmoid_derivative(k)
            rho = numpy.dot(self.weights[-l+1].transpose(), rho) * sd
            del_bias[-l] = rho
            del_weights[-l] = numpy.dot(rho, activations_list[-l-1].transpose())

        #print 'back propagation completed.....'
        #print ' debug:',del_bias
        #print 'debug2', del_weights
        #print 'done'
        return del_bias, del_weights


    def StochasticGradientDescent(self, train_inp, train_out, test_inp, test_outp, eta, iters, batch_size):
        # training data whether input or output is passed in form of list
        # 1. eta - learning rate
        # 2. iters - number of iterations
        # 3. batch_size - size of each batch into which train data is divided

        train_size = len(train_inp)
        bias_accumulate = [numpy.zeros(b.shape) for b in self.bias]
        weights_accumulate = [numpy.zeros(w.shape) for w in self.weights]
        # Performing iterations
        for one_iteration in xrange(iters):
            # making batches i.e dividing training data list into small batches i.e list of batches
            Batch_input = [train_inp[k:k+batch_size] for k in xrange(0, train_size, batch_size)]
            Batch_output = [train_out[k:k+batch_size] for k in xrange(0, train_size, batch_size)]
            #print 'Batch in:', Batch_input
            #print 'Batch out:', Batch_output

            for bin, bout in zip(Batch_input, Batch_output):   # picking the batch
                for train_in, train_ou in zip(bin, bout):     # picking the input and output from batch
                    # calling Back propagation
                    change_bias, change_weights = self.backpropagate(train_in, train_ou)
                    # accumulating the changes of all inputs within a batch
                    bias_accumulate = [b+delb for b, delb in zip(bias_accumulate, change_bias)]
                    weights_accumulate = [w+delw for w, delw in zip(weights_accumulate, change_weights)]

                # changing the biases and weights
                print 'length ',self.weights, (-eta/len(bin))*weights_accumulate[0]
                self.bias = [(b + ((-eta/len(bin)) * accumulated_delb)) for b, accumulated_delb in zip(self.bias, bias_accumulate ) ]
                self.weights = [(w + ((-eta/len(bout)) * accumulated_delw)) for w, accumulated_delw in zip(self.weights, weights_accumulate ) ]

            print 'bias:', self.bias
            print('weights:', self.weights)
            if test_inp:
                print "Iteration {0}:".format(one_iteration)
                self.evaluate(test_inp, test_outp)
            else:
                print "Iteration {0} complete". format(one_iteration)


    def TrainTest(self, train_inp, train_out,test_inp, test_outp, eta, iters, batch_size):
       self.StochasticGradientDescent(train_inp, train_out, test_inp,test_outp, eta, iters, batch_size)

    def evaluate(self, test_inp, test_outp):
        for test_item, test_item_out in zip(test_inp, test_outp):
            x = self.feedingforward(test_item)
            print 'x:', x
            print 'y:', test_item_out