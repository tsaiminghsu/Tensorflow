# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:10:40 2019

@author: USER
"""

import numpy as np
np.set_printoptions(threshold=np.inf)
import math
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance

par_dim = 2
par_num = 10
iter_num = 50
#w = 1
#c1 = 0.5
#c2 = 0.5
goal = np.random.rand(par_dim)

class Particle_Class:
    fit = 0
    pb_fit = 0
    x = np.array( [0.0]*par_dim )
    pb = np.copy(x)
    
    def get_fit(self):
            self.fit = 1.0/(1.0+distance.euclidean(self.x,goal))
            if self.fit > self.pb_fit:
                self.pb_fit = self.fit
                self.pb = np.copy(self.x)
        
class PSO_Class:
    gb_fit = 0
    gb = np.array( [0.0]*par_dim )
    particle = [Particle_Class() for i in range(par_num)]
    def __init__(self):
        for i in range(par_num):
            self.particle[i].x = np.random.rand(par_dim)
        self.gb = np.copy(self.particle[0].x)
    def get_all_fit(self):
        for i in range(par_num):
            self.particle[i].get_fit()
            if self.particle[i].pb_fit > self.gb_fit:
                self.gb_fit = self.particle[i].pb_fit
                self.gb = np.copy(self.particle[i].pb)
        
    def update(self):
        for i in range(par_num):
            self.particle[i].x =np.random.rand(par_dim)
            
    def plot_particles(self):
        plt.clf()
        for i in range(par_num):
          plt.scatter(self.particle[i].x[0],self.particle[i].x[1],color='blue',s=50,alpha=0.3,marker='o')
        plt.scatter(self.gb[0],self.gb[1],color='green',s=250,alpha=0.7,marker='+')  
        plt.scatter(goal[0],goal[1],color='red',s=250,alpha=1.0,marker='*')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.grid()
        plt.title('Iteration' + str(iteration) + ',Fitness: ' + str(self.gb_fit))
        plt.show()
        plt.pause(0.2)

PSO = PSO_Class()
for iteration in range(iter_num):
    PSO.get_all_fit()
    PSO.update()
    PSO.plot_particles()