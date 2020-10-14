import numpy as np
#import matplotlib.pyplot as plt
#from math_operations import *
from copy import deepcopy


class PCUnit():
    def __init__(self, dim=1, tau=0.02, bottom=False):
        '''
         u = PCUnit(dim=1, tau=0.02, bottom=False)

         Creates a PCUnit object.

         Inputs:
          dim   dimension of the internal v and e vectors (only tested for dim=1)
          tau   time constant for DE
          bottom  True if this is a bottom layer, otherwise Fasle
        '''
        self.dim = dim     # dimension of array (only tested for dim=1)
        self.tau = tau     # time constant for differential equations
        self.v = np.zeros(dim)   # value
        self.e = np.zeros(dim)   # error
        self.dv = np.zeros(dim)  # for computing the rate-of-change of v
        self.de = np.zeros(dim)  # ... and for e

        # this was a kluge for comparing to Millidge and Tschantz
        self.mu = np.zeros_like(self.v)
        self.fixed_mu = False   # turn this stuff off by default

        # We can clamp the value and error nodes individually.
        # If this is a "bottom" node, then the error node is
        # clamped by default.
        self.clamp_v = False
        self.bottom = bottom
        if self.bottom:
            self.clamp_e = True
        else:
            self.clamp_e = False

    def __repr__(self):
        '''
         print(u)
        '''
        return '[ e'+str(self.e)+' == v'+str(self.v)+' ]'

    def randomize_state(self, noise=1.):
        '''
         u.randomize_state(noise=1.)

         Sets v and e nodes to random values using a Normal
         distribution with stdev set by the "noise" parameter.
        '''
        self.v = np.random.normal(scale=noise, size=self.v.shape)
        self.e = np.random.normal(scale=noise, size=self.e.shape)

    def v_receive(self, x):
        '''
         u.v_receive(x)
         Increments the input to the value node by x.
        '''
        self.dv += x

    def e_receive(self, x):
        '''
         u.e_receive(x)
         Increments the input to the error node by x.
        '''
        self.de += x

    def fix_mu(self):
        '''
         Only used to compare to M&T.
        '''
        self.fixed_mu = True
        self.mu = deepcopy(self.v)

    def step(self, dt=0.001):
        '''
         u.step(dt=0.001)

         Takes an Euler step of length dt, and updates the internal
         value and error nodes according to the derivatives that
         have been accumulated since the last step (in u.dv and u.de).

         This method also resets the derivative accumulators.
        '''
        # These next two lines implement the INTERNAL activity between
        # the value and error nodes.
        self.dv -= self.e
        self.de += self.v - self.e

        # Euler step
        if not self.clamp_v:
            # Only update if not clamped
            self.v += dt/self.tau * self.dv
        if not self.clamp_e:
            # Only update if not clamped
            self.e += dt/self.tau * self.de
        self.dv = np.zeros_like(self.v)  # reset to 0
        self.de = np.zeros_like(self.e)  # reset to 0
