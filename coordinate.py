import numpy as np
import time
import car
import math
import dynamics
import matplotlib.cm
import theano as th
import utils
import feature
import pickle
import sys
from car import Car
import midlevel

class Coordinate(object):
    
    def __init__(self):
        self.lanes = []
        self.cars = []
        self.prev_x = {}
        self.feed_u = None
        self.feed_x = None
        self.prev_t = None

        self.midlevel_exists = False
        self.midlevel = None
        # number of times the control loop should run per midlevel run
        self.cntrl_loop_per_midlevel_iter = 30 
        # number of times the control loop has run since last midlevel run
        self.cntrl_loop_counter = 0
        self.termination_counter = 0
        self.MAX_ITER = 200 
        self.aut_level = None

    def use_world(self, world):
        self.cars = [c for c in world.cars]
        self.lanes = [c for c in world.lanes]
        self.objects = [c for c in world.objects]

        self.aut_level = world.aut_level

        # add midlevel optimizer if it exists
        self.midlevel_exists = world.midlevel_exists
        if self.midlevel_exists:
            # Populate the midlevel optimizer
            # TODO: fix world.road call here
            self.midlevel = midlevel.Midlevel(world, world.lanes, self.cars)

    # this function calls the midlevel optimizer at a fixed frequency
    def midlevel_loop(self, _):
        # Only run every dt_midlevel seconds of actual simulation.
        # If the control loop is running very slowly, run midlevel accordingly slowly
        if self.cntrl_loop_counter >= self.cntrl_loop_per_midlevel_iter:
            self.cntrl_loop_counter = 0
            self.midlevel.increment_goals()

    def control_loop(self):
        steer, gas = 0,0
        if self.feed_u is None:
            for car in reversed(self.cars):
                car.control(steer, gas)
        else:
            for car, fu, hu in zip(self.cars, self.feed_u, self.history_u):
                car.u = fu[len(hu)]
        for car, hist in zip(self.cars, self.history_u):
            hist.append(car.u)
        for car in self.cars:
            self.prev_x[car] = car.x
        for car in self.cars:
            print("CAR: {}".format(car.x))
            car.move()
        for car, hist in zip(self.cars, self.history_x):
            hist.append(car.x)
        self.prev_t = time.time()
        self.termination_counter += 1
        if self.termination_counter >= self.MAX_ITER:
            data2_filename = "data2/CDC_RUN_NUM_CARS{0}_AUTLEVEL{1}_TIME{2}.txt".\
                    format(len(self.cars), self.aut_level, time.time())
            write_string = ""
            for car in self.cars:
                cur_write = {'lane': car.x[0], 'pos': car.x[1], 'isRobot': car.iamrobot}
                write_string += str(cur_write)
                write_string += "\n"
            
            with open(data2_filename, 'w') as f:
                f.write(write_string)
        #increment the counter of how many control loop iterations since the last midlevel iteration
        self.cntrl_loop_counter += 1

    def reset(self):
        for car in self.cars:
            car.reset()
        self.prev_t = time.time()
        for car in self.cars:
            self.prev_x[car] = car.x
        self.paused = True
        self.history_x = [[] for car in self.cars]
        self.history_u = [[] for car in self.cars]

    def run(self):
        self.reset()
        for i in range(self.MAX_ITER):
            self.control_loop()
            if i % 30 == 0 and self.midlevel_exists:
                self.midlevel_loop()


