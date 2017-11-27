import theano
from theano import theano as T

import numpy

class HiddenLayer:
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
                
            W_branches = theano.shared(value=W_values, name='W_branches', borrow=True)
                
            if b is None:
                    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
                    b_1 = theano.shared(value=b_values, name='b_1', borrow=True)

        self.W_branches = W_branches
        self.b_1 = b_1

        sub_branch_type = ""
        z_i = T.concatenate(self.W_branches[sub_branch_type] + self.W_branches[sub_branch_dist])
        
        # self.output = 

        self.params = [self.W_branches, self.b_1]


class DeepVS:
    def __init__(self, rng, input, n_in, n_hidden, n_conv, n_out):
        """Initialize the parameters for the multilayer perceptron
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie
        
        :type n_hidden: int
        :param n_hidden: number of hidden units
        
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        
        """
        self.hidden_layer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.errors = self.convolutional_layer.errors

        self.params = self.hidden_layer.params + self.convolutional_layer.params


def train_model(learning_rate=0.1,
                n_epochs=1000,
                batch_size=20,
                n_hidden=500):

    # BUILD MODEL #
    print("...building model")

    rng = numpy.random.RandomState(1234)

    classifier = DeepVS(
        rng=rng,
        input=x,
        n_in=42,
        n_hidden=n_hidden,
        n_out=3
    )

    cost = (
            classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
    )

    test_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
