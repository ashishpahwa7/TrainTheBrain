__author__ = 'h_hack'
import trainbrain
import numpy

def main():
    mynet = trainbrain.NeuralNetwork([3, 1])
    print 'weights:', mynet.weights
    print 'bias:', mynet.bias
    print 'layers:', mynet.layers
    v1= numpy.array([[0], [0], [1]])
    v2 = numpy.array([[0], [1], [0]])
    v3 = numpy.array([[1], [0], [0]])
    v15 = numpy.array([[1], [0], [0]])
    v16 = numpy.array([[1], [0], [0]])
    v17 = numpy.array([[1], [0], [0]])
    v4 = numpy.array([[1], [1], [0]])
    v12 = numpy.array([[1], [1], [0]])
    v13 = numpy.array([[1], [1], [0]])
    v14 = numpy.array([[1], [1], [0]])
    v5 = numpy.array([[1], [1], [1]])
    v6 = numpy.array([[1], [1], [1]])
    v7 = numpy.array([[0], [0], [0]])
    v8 = numpy.array([[0], [0], [0]])
    v9 = numpy.array([[0], [0], [0]])
    v10 = numpy.array([[0], [0], [0]])
    v11 = numpy.array([[0], [0], [0]])

    o1 = numpy.array([1])

    o2 = numpy.array([1])

    o3 = numpy.array([1])
    o15 = numpy.array([1])
    o16 = numpy.array([1])
    o17 = numpy.array([1])

    o4 = numpy.array([0])
    o12 = numpy.array([0])
    o13 = numpy.array([0])
    o14 = numpy.array([0])

    o5 = numpy.array([1])
    o6 = numpy.array([1])

    o7 = numpy.array([0])
    o8 = numpy.array([0])
    o9 = numpy.array([0])
    o10 = numpy.array([0])
    o11 = numpy.array([0])

    test1 = numpy.array([[1], [1], [1]])
    test2 = numpy.array([[1], [0], [0]])
    test3 = numpy.array([[0], [1], [1]])
    test4 = numpy.array([[0], [0], [0]])
    testo1 = numpy.array([1])
    testo2 = numpy.array([1])
    testo3 = numpy.array([0])
    testo4 = numpy.array([0])
    train_inp = [v1, v2, v3, v15, v16, v17, v4, v12, v13, v14, v5, v6, v7, v8, v9, v10, v11 ]
    train_out = [o1, o2, o3, o15, o16, o17, o4, o12, o13, o14, o5, o6, o7, o8, o9, o10, o11]
    test_inp = [test1, test2, test3, test4]
    test_outp = [testo1, testo2, testo3, testo4]
    #ynet.feedingforward(v1)
    mynet.TrainTest(train_inp, train_out, test_inp, test_outp, 0.25, 10000, 4)

if __name__ == '__main__':
    main()
