import numpy as np
#import matplotlib.pyplot as plt
#from math_operations import *
from copy import deepcopy

class PCConx(object):
    def __init__(self, pre, post, func=None):
        '''
         conx = PCConx(pre, post, funct=None)

         Creates a connection (object) between two PCUnits.

            pre    func   post
         [(e)=(v)] <==> [(e)=(v)]

         Inputs:
          pre     a PCUnit object
          post    a PCUnit object
          func    a function object
        '''
        self.pre = pre
        self.post = post
        self.func = func

    def send(self):
        '''
         conx.send()

         Computes the activity sent between PCUnits.
         Note that this function addresses only the connections
         BETWEEN PCUnits, and not the connections internal to the units.

            pre    func   post
         [(e)=(v)] <==> [(e)=(v)]
        '''
        if self.post.fixed_mu:
            # This code is just to compare to M&T
            self.post.e_receive(-self.post.mu)
            self.pre.v_receive(self.post.e*self.func.D(self.pre.mu))
        else:
            # This is the code that implements PC dynamics
            # pre_v ==> post.e
            self.post.e_receive(-self.func(self.pre.v))
            # pre_v <== post.e
            self.pre.v_receive(self.post.e*self.func.D(self.pre.v))
