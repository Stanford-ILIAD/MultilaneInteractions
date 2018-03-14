#!/usr/bin/env python2.7
import sys
import coordinate
import world
import theano as th
from car import UserControlledCar
#import gc

th.config.optimizer_verbose = False 
th.config.allow_gc = False

if __name__ == '__main__':
    world_to_use = world.world7(0.1)
    coord = coordinate.Coordinate()
    coord.use_world(world_to_use)
    coord.run()
    #gc.collect()
    print("HERE!")
    del world_to_use
    del coord
    world_to_use = world.world7(0.3)
    coord = coordinate.Coordinate()
    coord.use_world(world_to_use)
    coord.run()
    print("HERE AGAIN!")
    
