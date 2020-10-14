import numpy as np

class logistic(object):
    @staticmethod
    def __call__(x):
        '''
         The __call__ method is the one that gets called by default.

         If you had the code
         >>> func = logistic()
         then you could call the function using
         >>> func(3.6)
        '''
        return 1./(1.+np.exp(-x))
    @classmethod
    def D(cls, x):
        '''
         This is the function that computes the derivative.

         Call it using
         >>> func.D(3.6)
        '''
        y = cls.__call__(x)
        return y*(1.-y)
    @classmethod
    def __repr__(cls):
        '''
         The __repr__ method is called when you "print" your object.

         Call it using
         >>> print(func)
        '''
        return 'logistic'

class identity(object):
    @staticmethod
    def __call__(x):
        return x
    @classmethod
    def D(cls, x):
        return np.ones_like(x)
    @classmethod
    def __repr__(cls):
        return 'identity'

class scalarmult(object):
    def __init__(self, c):
        '''
         func = scalarmult(c)

         Note that this class requires an input scalar value.
        '''
        self.c = c
    def __call__(self, x):
        return self.c*x
    def D(self, x):
        return self.c*np.ones_like(x)
    def __repr__(self):
        return 'multiply by '+str(self.c)

class scaledlogistic(object):
    def __init__(self, c):
        '''
         func = scaledlogistic(c)

         Note that this class requires an input scalar value.
        '''
        self.c = c
    def __call__(self, x):
        return self.c/(1.+np.exp(-x))
    def D(self, x):
        y = self.__call__(x) / self.c
        return y*(1.-y)*self.c
    def __repr__(self):
        return 'logistic, scaled by '+str(self.c)

class absval(object):
    @staticmethod
    def __call__(x):
        return np.abs(x)
    @classmethod
    def D(cls, x):
        return np.sign(x)
    @classmethod
    def __repr__(cls):
        return 'absval'

class tanh(object):
    @staticmethod
    def __call__(x):
        return np.tanh(x)
    @classmethod
    def D(cls, x):
        y = cls.__call__(x)
        return 1 - y**2
    @classmethod
    def __repr__(cls):
        return 'tanh'

class scaledtanh(object):
    def __init__(self, c):
        self.c = c
    def __call__(self, x):
        return self.c*np.tanh(x)
    def D(self, x):
        y = self.__call__(x) / self.c
        return (1.-y**2)*self.c
    def __repr__(self):
        return 'tanh, scaled by '+str(self.c)

class tan(object):
    @staticmethod
    def __call__(x):
        return np.tan(x)
    @classmethod
    def D(cls, x):
        return 1. / (np.cos(x)**2)
    @classmethod
    def __repr__(cls):
        return 'tan'

class sqrt(object):
    @staticmethod
    def __call__(x):
        return np.sqrt(np.abs(x))
        #return np.sqrt(x)
    @classmethod
    def D(cls, x):
        return 0.5/np.sqrt(np.abs(x))*np.sign(x)
        #return 0.5/np.sqrt(x)
    @classmethod
    def __repr__(cls):
        return 'sqrt'

class square(object):
    @staticmethod
    def __call__(x):
        return x**2
    @classmethod
    def D(cls, x):
        return 2.*x
    @classmethod
    def __repr__(cls):
        return 'square'

class sin(object):
    @staticmethod
    def __call__(x):
        return np.sin(x)
    @classmethod
    def D(cls, x):
        y = cls.__call__(x)
        return np.cos(x)
    @classmethod
    def __repr__(cls):
        return 'sin'
