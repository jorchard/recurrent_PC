import numpy as np
import matplotlib.pyplot as plt
from math_operations import *
from PCUnit import *
from PCConx import *
from copy import deepcopy

class PCNet(object):
    def __init__(self):
        '''
         net = PCNet()
        '''
        self.unit = []   # list of PCUnit
        self.conx = []   # list of PCConx
        self.time = 0.   # current simulation time

    def __repr__(self):
        s = ''
        for u in self.unit:
            s = s + str(u)+'\n'
        return s

    def randomize_state(self, noise=1.):
        '''
         net.randomize_state(noise=1.)

         Randomly sets the error and value nodes to random values
         using a Normal distribution with stdev = "noise".
        '''
        for u in self.unit:
            u.randomize_state(noise=noise)

    def fix_mu(self):
        '''
         Only used to compare to M&T.
        '''
        for u in self.unit:
            u.fix_mu()
    def release_mu(self):
        for u in self.unit:
            u.fixed_mu = False

    def set_tau(self, tau):
        '''
         net.set_tau(tau)

         Sets the time constant for the PCUnits.
        '''
        for u in self.unit:
            u.tau = tau

    def add_unit(self, u):
        '''
         net.add_unit(u)

         Adds a PCUnit to the network.
        '''
        self.unit.append(u)

    def connect(self, a, b, func=None):
        '''
         net.connect(a, b, func=None)

         Connects two PCUnits that are already in the network.
        '''
        c = PCConx(a, b, func=func)
        print(c.func)
        self.conx.append(c)

    def integrate(self):
        '''
         net.integrate()

         Executes the exhange of current between PCUnits.
        '''
        for c in self.conx:
            c.send()

    def step(self, dt=0.001):
        '''
         net.step(dt=0.001)

         Takes an Euler step of length dt. This method updates the
         PCUnits according to their accumulated input current.
        '''
        for u in self.unit:
            u.step(dt=dt)

    def run(self, T, dt=0.001):
        '''
         ts, us = net.run(T, dt=0.001)

         Runs the PC net for T seconds, taking Euler steps of length dt.

         Outputs:
          ts    time stamps (array of length N)
          us    a list with one entry per PCUnit
                Each entry has the form:
                2 x N x dim, such that
                us[i][node][n][0] is the
                i-th PCUnit
                node is 0 for error, and 1 for value
                n is the n-th time step
        '''
        t = np.arange(self.time, self.time+T, step=dt)
        P = len(t)
        u_hist = []
        for u in self.unit:
            u_hist.append([np.zeros((P,*np.shape(u.e))), np.zeros((P,*np.shape(u.v)))])
        #self.record()
        for k,tt in enumerate(t):
            self.integrate()
            self.step(dt=dt)
            self.time = tt
            for u_idx,u in enumerate(u_hist):
                u[0][k] = deepcopy(self.unit[u_idx].e)
                u[1][k] = deepcopy(self.unit[u_idx].v)
            #self.record()
        return t, u_hist
