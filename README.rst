.. raw:: html

    <div align="center">
      <img src="https://raw.githubusercontent.com/MikhailKravets/Spytnik/master/logo.jpg"><br><br>
    </div>

.. image:: https://travis-ci.org/MikhailKravets/Spytnik.svg?branch=master
    :target: https://travis-ci.org/MikhailKravets/Spytnik

.. image:: https://codecov.io/gh/MikhailKravets/Spytnik/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/MikhailKravets/Spytnik

Spytnik is the Python project that gathers modern deep learning algorithms; the field of experiments
and place for application of new knowledge.

Usage example
*************

Let's try to approximate very simple function `0.5*sin(e^x) - cos(e^(-x))`.

.. code:: python

    # import all we need
    import matplotlib.pyplot as plot
    import random
    
    import numpy
    
    # that is our project's packages. Mehh... it's not documented yet :(
    import layers
    from core import FeedForward, separate_data
    from core.estimators import cv
    
    
    # the very simple function
    def f(x):
        return 0.5 * numpy.sin(numpy.exp(x)) - numpy.cos(numpy.exp(-1 * x))
    
    
    # creating new FeedForward instance.
    # FeedForward is our neural netrowk which we'll be trained
    nn = FeedForward(momentum=0.2, learn_rate=0.05, weight_decay=0.2)
    
    # Append layers to neural networks
    # That's an interesting moment:
    # You can combine any layers you want. There is only one constraint: input layer must
    # have the amount of neurons equal to input vector length; such as output layer must
    # have the amount of neurons equal to output vector length.
    #
    # Such way, here we create one linear input layer (arguments (1, 10) means that the layer contains
    # 1 layer and 10 synapsis are outputed from the each neuron in layer);
    #
    # three hidden layer with Tanh activation function;
    # and one linear output layer 
    nn += layers.Linear(1, 10)
    nn += layers.Tanh(10, 10)
    nn += layers.Tanh(10, 10)
    nn += layers.Tanh(10, 1)
    nn += layers.Linear(1, 0)
    
    # generate input data and desired output to this data
    data = [([x], [f(x)]) for x in numpy.linspace(-2.2, 2.5, 150)]
    
    # separate it on the training set and validation ses
    ts, vs = separate_data(data, 0.15)
    
    
    # duplicate x and y for easier plotting
    x = numpy.linspace(-2.2, 2.5, 150)
    y = f(x)
    
    # train the neural networks feeding it the elements from training set randomly
    error = []
    v_error = []
    for i in range(50_000):
        r = random.randint(0, len(ts) - 1)
        
        # here we train it... step by step...
        nn.fit(ts[r][0], ts[r][1])
        
        # just to see on the errors further
        error.append(nn.error)
        if i % 300 == 0:
            v_error.append(cv(nn, vs))
    
    # use our trained neural network for approximation of our simple function
    y_trained = []
    for v in x:
        y_trained.append(nn.get([v])[0])
    
    # just plot the result
    plot.subplot(211)
    plot.title("f(x) and its approximation")
    plot.plot(x, y)
    plot.plot(x, y_trained)
    
    plot.subplot(212)
    plot.title("Learning error")
    plot.plot(error)
    plot.plot([i * 300 for i in range(len(v_error))], v_error)
    plot.show()

Plotted charts after running of the code above should be similar to this one ↓

.. raw:: html

    <div align="center">
      <img src="https://raw.githubusercontent.com/MikhailKravets/Spytnik/master/doc/fig1.png"><br><br>
    </div>

Awesome, yeah?

There will be some more awesome information in the readme or even personal website but some later, wait a little.

Documentation and so on
***********************

I will write documentation with mathematical background to it, I promise... just believe me

License
*******

MIT License

Attribution
***********

The red sputnik from logo is made by `Freepik <https://www.freepik.com/>`_
