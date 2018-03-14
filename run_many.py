#!/usr/bin/env python2.7
import sys
import visualize
import world
import theano as th
from car import UserControlledCar
#import gc

th.config.optimizer_verbose = False 
th.config.allow_gc = False

if __name__ == '__main__':
    name = sys.argv[1]
    if len(sys.argv)>2 and sys.argv[2]=='fast':
        th.config.optimizer = 'fast_compile'
    if len(sys.argv)>2 and sys.argv[2]=='FAST':
        th.config.mode = 'FAST_COMPILE'
    world_to_use = getattr(world, name)()
    if len(sys.argv)>3 or (len(sys.argv)>2 and sys.argv[2] not in ['fast', 'FAST']):
        ctrl = eval(sys.argv[-1])
        for car in world.cars:
            if isinstance(car, UserControlledCar):
                print 'User Car'
                car.fix_control(ctrl)
    vis = visualize.Visualizer(0.1, name=name)
    vis.use_world(world_to_use)
    vis.main_car = world_to_use.cars[0]
    vis.run()
    del vis
    del world_to_use
    #gc.collect()
    print("HERE!")
    world_to_use = getattr(world, name)()
    vis = visualize.Visualizer(0.1, name=name)
    vis.use_world(world_to_use)
    vis.main_car = world_to_use.cars[0]
    vis.run()
    print("HERE AGAIN!")
