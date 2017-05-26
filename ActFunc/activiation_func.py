'''
For initial commit of general activation function. 
'''
import tensorflow as tf
import numpy as np 
from tensorflow.python.framework import ops


def generate_function(x_range=np.array([0.0,1.0], dtype=float32), # range of function  
                      y_range=(0,1), # domain of function  
                      extend_function = False, # extend function beyond domain
                      is_continuos = True, # return continuos function
                      n_interval = 100, # number of intervals
                      ):
    '''
    Return a generator of tuple of (function, derivative) for given input. 
    
    -------------------------------
    parameters
    x_range : tuple
    
    y_range : tuple
    
    extend_function : bool 
    
    is_continuos : bool 
    
    '''
    x0, x1 = x_range
    y0, y1 = y_range
    interval = (x1-x0)/n_interval
    
    
    
    def _lin(x):
        pass
    
    # derivative function generation
    def d_func(x):
        pass
        
    # function for f
    def func(x):
        pass
    
    def d_spiky(x):
        r = x % 1
        if r <= 0.5:
            return 1
        else:
            return 0

    def spiky(x):
        r = x % 1
        if r <= 0.5:
            return r
        else:
            return 0
            
    np_spiky = np.vectorize(spiky)
    np_d_spiky = np.vectorize(d_spiky)    
    yield (np_spiky, np_d_spiky)
    
    
    
for np_spiky, np_d_spiky in generate_function():
    def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

    def tf_d_spiky(x,name=None):
        with ops.op_scope([x], name, "d_spiky") as name:
            y = tf.py_func(np_d_spiky_32,
                            [x],
                            [tf.float32],
                            name=name,
                            stateful=False)
            return y[0]
            
    np_spiky_32 = lambda x: np_spiky(x).astype(np.float32)
    np_d_spiky_32 = lambda x: np_d_spiky(x).astype(np.float32)

    def spikygrad(op, grad):
        x = op.inputs[0]

        n_gr = tf_d_spiky(x)
        return grad * n_gr  
        
        

    def tf_spiky(x, name=None):
        with ops.op_scope([x], name, "spiky") as name:
            y = py_func(np_spiky_32,
                            [x],
                            [tf.float32],
                            name=name,
                            grad=spikygrad)  # <-- here's the call to the gradient
            return y[0]

    

    with tf.Session() as sess:

        x = tf.constant([0.2,0.7,1.2,1.7])
        y = tf_spiky(x)
        tf.initialize_all_variables().run()

        print(x.eval(), y.eval(), tf.gradients(y, [x])[0].eval())