import lane
import car
import math
import feature
import dynamics
import visualize
import utils
import sys
import theano as th
import theano.tensor as tt
import numpy as np
import shelve

th.config.optimizer_verbose = False
th.config.allow_gc = False
th.config.optimizer = 'fast_compile'

class Object(object):
    def __init__(self, name, x):
        self.name = name
        self.x = np.asarray(x)

class World(object):
    def __init__(self, midlevel_exists = False):
        self.cars = []
        self.lanes = []
        self.roads = []
        self.fences = []
        self.objects = []
        self.midlevel_exists = midlevel_exists
        self.aut_level = None
    def simple_reward(self, trajs=None, lanes=None, roads=None, fences=None, speed=1., speed_import=1.):
        if lanes is None:
            lanes = self.lanes
        if roads is None:
            roads = self.roads
        if fences is None:
            fences = self.fences
        if trajs is None:
            trajs = [c.linear for c in self.cars]
        elif isinstance(trajs, car.Car):
            trajs = [c.linear for c in self.cars if c!=trajs]
        r = 0.1*feature.control()
        theta = [1., -50., 10., 10., -60.] # Simple model
        # theta = [.959, -46.271, 9.015, 8.531, -57.604]
        for lane in lanes:
            r = r+theta[0]*lane.gaussian()
        for fence in fences:
            r = r+theta[1]*fence.gaussian()
        for road in roads:
            r = r+theta[2]*road.gaussian(10.)
        if speed is not None:
            r = r+speed_import*theta[3]*feature.speed(speed)
        for traj in trajs:
            r = r+theta[4]*traj.gaussian()
        return r

def playground():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.17)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    #world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.], color='orange'))
    world.cars.append(car.UserControlledCar(dyn, [-0.17, -0.17, math.pi/2., 0.], color='white'))
    return world

def irl_ground():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    d = shelve.open('cache', writeback=True)
    cars = [(-.13, .1, .5, 0.13),
            (.02, .4, .8, 0.5),
            (.13, .1, .6, .13),
            (-.09, .8, .5, 0.),
            (0., 1., 0.5, 0.),
            (-.13, -0.5, 0.9, 0.13),
            (.13, -.8, 1., -0.13),
           ]
    def goal(g):
        @feature.feature
        def r(t, x, u):
            return -(x[0]-g)**2
        return r
    for i, (x, y, s, gx) in enumerate(cars):
        if str(i) not in d:
            d[str(i)] = []
        world.cars.append(car.SimpleOptimizerCar(dyn, [x, y, math.pi/2., s], color='yellow'))
        world.cars[-1].cache = d[str(i)]
        def f(j):
            def sync(cache):
                d[str(j)] = cache
                d.sync()
            return sync
        world.cars[-1].sync = f(i)
    for c, (x, y, s, gx) in zip(world.cars, cars):
        c.reward = world.simple_reward(c, speed=s)+10.*goal(gx)
    world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.7], color='red'))
    world.cars = world.cars[-1:]+world.cars[:-1]
    return world

def world_test():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    world.cars[1].reward = world.simple_reward(world.cars[1], speed=0.5)
    return world

def world0():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human_speed(t, x, u):
        return -world.cars[1].traj_h.x[t][3]**2
    r_r = world.simple_reward(world.cars[1], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    return world

def world1(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    r_h = world.simple_reward([world.cars[1].traj], speed_import=.2 if flag else 1., speed=0.8 if flag else 1.)+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human_speed(t, x, u):
        return -world.cars[1].traj_h.x[t][3]**2
    r_r = 300.*human_speed+world.simple_reward(world.cars[1], speed=0.5)
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    world.cars[1].rewards = (r_h, r_r)
    #world.objects.append(Object('cone', [0., 1.8]))
    return world

def world2(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[0].bounds = [(-3., 3.), (-1., 1.)]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return -(world.cars[1].traj_h.x[t][0])*10
    r_r = 300.*human+world.simple_reward(world.cars[1], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    #world.objects.append(Object('firetruck', [0., 0.7]))
    return world

def world3(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    world.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[0].bounds = [(-3., 3.), (-1., 1.)]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    r_h = world.simple_reward([world.cars[1].traj])+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return (world.cars[1].traj_h.x[t][0])*10
    r_r = 300.*human+world.simple_reward(world.cars[1], speed=0.5)
    world.cars[1].rewards = (r_h, r_r)
    #world.objects.append(Object('firetruck', [0., 0.7]))
    return world

def world4(flag=False):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    vlane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    hlane = lane.StraightLane([-1., 0.], [1., 0.], 0.13)
    world.lanes += [vlane, hlane]
    world.fences += [hlane.shifted(-1), hlane.shifted(1)]
    world.cars.append(car.UserControlledCar(dyn, [0., -.3, math.pi/2., 0.0], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [-0.3, 0., 0., 0.], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[0].bounds = [(-3., 3.), (-2., 2.)]
    if flag:
        world.cars[0].follow = world.cars[1].traj_h
    world.cars[1].bounds = [(-3., 3.), (-2., 2.)]
    @feature.feature
    def horizontal(t, x, u):
        return -x[2]**2
    r_h = world.simple_reward([world.cars[1].traj], lanes=[vlane], fences=[vlane.shifted(-1), vlane.shifted(1)]*2)+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return -tt.exp(-10*(world.cars[1].traj_h.x[t][1]-0.13)/0.1)
    r_r = human*10.+horizontal*30.+world.simple_reward(world.cars[1], lanes=[hlane]*3, fences=[hlane.shifted(-1), hlane.shifted(1)]*3+[hlane.shifted(-1.5), hlane.shifted(1.5)]*2, speed=0.9)
    world.cars[1].rewards = (r_h, r_r)
    return world

def world5():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    vlane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    hlane = lane.StraightLane([-1., 0.], [1., 0.], 0.13)
    world.lanes += [vlane, hlane]
    world.fences += [hlane.shifted(-1), hlane.shifted(1)]
    world.cars.append(car.UserControlledCar(dyn, [0., -.3, math.pi/2., 0.0], color='red'))
    world.cars.append(car.NestedOptimizerCar(dyn, [-0.3, 0., 0., 0.0], color='yellow'))
    world.cars[1].human = world.cars[0]
    world.cars[1].bounds = [(-3., 3.), (-2., 2.)]
    @feature.feature
    def horizontal(t, x, u):
        return -x[2]**2
    r_h = world.simple_reward([world.cars[1].traj], lanes=[vlane], fences=[vlane.shifted(-1), vlane.shifted(1)]*2)+100.*feature.bounded_control(world.cars[0].bounds)
    @feature.feature
    def human(t, x, u):
        return -tt.exp(10*(world.cars[1].traj_h.x[t][1]-0.13)/0.1)
    r_r = human*10.+horizontal*2.+world.simple_reward(world.cars[1], lanes=[hlane]*3, fences=[hlane.shifted(-1), hlane.shifted(1)]*3+[hlane.shifted(-1.5), hlane.shifted(1.5)]*2, speed=0.9)
    world.cars[1].rewards = (r_h, r_r)
    return world

def world6(know_model=True):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2), clane.shifted(2.5), clane.shifted(-2.5)]
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 0.5], color='red'))
    if know_model:
        world.cars.append(car.NestedOptimizerCar(dyn, [0., 0.05, math.pi/2., 0.5], color='yellow'))
    else:
        world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.05, math.pi/2., 0.5], color='yellow'))
    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.6)
    world.cars[0].default_u = np.asarray([0., 1.])
    @feature.featureq
    def goal(t, x, u):
        return -(10.*(x[0]+0.13)**2+0.5*(x[1]-2.)**2)
    if know_model:
        world.cars[1].human = world.cars[0]
        r_h = world.simple_reward([world.cars[1].traj], speed=0.6)+100.*feature.bounded_control(world.cars[0].bounds)
        r_r = 10*goal+world.simple_reward([world.cars[1].traj_h], speed=0.5)
        world.cars[1].rewards = (r_h, r_r)
    else:
        r = 10*goal+world.simple_reward([world.cars[0].linear], speed=0.5)
        world.cars[1].reward = r
    return world




# For testing the midlevel optimizer
def world7(prob_aut=0.5):
    num_cars, prob_aut = 20, prob_aut 
    aut_list = np.zeros(num_cars)
    dyn = dynamics.CarDynamics(0.1)
    world = World(midlevel_exists=True)
    world.aut_level = prob_aut
    clane = lane.StraightLane([0., -1.], [0., 1.], lane.DEFAULT_WIDTH)
    world.lanes += [clane, clane.shifted(1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-1), clane.shifted(2.5), clane.shifted(-1.5)]
   # world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 0.5], color='yellow'))
   # world.cars.append(car.SwitchOptimizerCar(dyn, [0, 0.15, math.pi/2., 0.5], color='red', iamrobot=True))
   # world.cars.append(car.SwitchOptimizerCar(dyn, [0, -0.3, math.pi/2., 0.5], color='red', iamrobot = True))
    lane_1_cur_pos = -10
    lane_2_cur_pos = -10.5
    for i in xrange(num_cars):
        temp_pos = None 
        temp_lane = np.random.binomial(1, 0.5)*(-0.13)
        if temp_lane < 0:
            temp_pos = lane_1_cur_pos
            lane_1_cur_pos += np.random.uniform(0.25, 0.5)
        else:
            temp_pos = lane_2_cur_pos
            lane_2_cur_pos += np.random.uniform(0.25, 0.5)
        if np.random.random() <= prob_aut:
            world.cars.append(car.SwitchOptimizerCar(dyn, [temp_lane, temp_pos,
                math.pi/2., 0.5], color='red', iamrobot=True))
            aut_list[i] = 1
        else:
            world.cars.append(car.SimpleOptimizerCar(dyn, [temp_lane, temp_pos,
                math.pi/2., 0.5], color='yellow'))
    
    print("\n\n\n\n\n\n\n\n\n\n\n\nAUT LIST IS: {} \n\n\n\n".format(aut_list))
    
   # world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.6)
   # world.cars[0].default_u = np.asarray([0., 1.])
   # world.cars[1].baseline_reward = world.simple_reward(world.cars[1], speed=0.6) 
   # world.cars[1].default_u = np.asarray([0., 1.])
   # world.cars[2].baseline_reward = world.simple_reward(world.cars[2], speed=0.6) 
   # world.cars[2].default_u = np.asarray([0., 1.])
   # world.cars[3].reward = world.simple_reward(world.cars[3], speed=0.6)
   # world.cars[3].default_u = np.asarray([0., 1.])
    for i in xrange(num_cars):
        if aut_list[i] == 1:
            world.cars[i].baseline_reward = world.simple_reward(world.cars[i], speed=0.6) 
            world.cars[i].default_u = np.asarray([0., 1.])
        else:
            world.cars[i].reward = world.simple_reward(world.cars[i], speed=0.6)
            world.cars[i].default_u = np.asarray([0., 1.])


    return world

# Start world in phase 1
def world8():
    dyn = dynamics.CarDynamics(0.1)
    world = World(midlevel_exists=True)
    clane = lane.StraightLane([0., -1.], [0., 1.], lane.DEFAULT_WIDTH)
    world.lanes += [clane, clane.shifted(1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-1), clane.shifted(2.5), clane.shifted(-1.5)]
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 0.5], color='yellow'))
    world.cars.append(car.SwitchOptimizerCar(dyn, [-0.13, 0.3, math.pi/2., 0.5], color='red', iamrobot=True))
    world.cars.append(car.SwitchOptimizerCar(dyn, [-0.13, -0.3, math.pi/2., 0.5], color='red', iamrobot = True))

    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.6)
    world.cars[0].default_u = np.asarray([0., 1.])
    world.cars[1].baseline_reward = world.simple_reward(world.cars[1], speed=0.6) 
    world.cars[1].default_u = np.asarray([0., 1.])
    world.cars[2].baseline_reward = world.simple_reward(world.cars[2], speed=0.6) 
    world.cars[2].default_u = np.asarray([0., 1.])

    return world


# World with one purely autonomous lane and one mixed lane
def world9():
    dyn = dynamics.CarDynamics(0.1)
    world = World(midlevel_exists=True)
    clane = lane.StraightLane([0., -1.], [0., 1.], lane.DEFAULT_WIDTH)
    world.lanes += [clane, clane.shifted(1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-1), clane.shifted(2.5), clane.shifted(-1.5)]
    car_set = [{'lane': -0.14382104671193202, 'pos': -5.4614089828930892, 'isRobot': True}, {'lane': -0.076485355678699476, 'pos': -5.287241227862939,'isRobot': True},\
            {'lane': 0.041717183967274571, 'pos': -5.1860610599902248,'isRobot': True},\
            {'lane': -0.13005408356976389, 'pos': -5.6439957056000463, 'isRobot': True},\
            {'lane': -0.0017838032137736108, 'pos': 1.900418486064837, 'isRobot':False},\
            {'lane': -0.1255197374127382, 'pos': 2.9384294124012125, 'isRobot':False},\
            {'lane': -0.0017836856576096698, 'pos': 2.2542116035764277,'isRobot': False},\
            {'lane': -0.0017545585415078939, 'pos': 3.5441324751687593,'isRobot': False},\
            {'lane': -0.15426277999902296, 'pos': -5.2395293216772982,'isRobot': True},\
            {'lane': -0.0016715524262676368, 'pos': 3.1864204004814645,'isRobot': False},\
            {'lane': -0.078191570164785756, 'pos': -5.4227005075263781,'isRobot': True},\
            {'lane': -0.17759460092156229, 'pos': -5.0843811548205187,'isRobot': True},\
            {'lane': -0.0017836890619227796, 'pos': 4.1049603048534156,'isRobot': False},\
            {'lane': -0.0017781598554608973, 'pos': 4.5832853862108536,'isRobot': False},\
            {'lane': -0.12587224415597753, 'pos': 3.3265390539200665,'isRobot':False},\
            {'lane': -0.043776679064340018, 'pos': -5.1047807513629015,'isRobot': True},\
            {'lane': -0.12555872982148231, 'pos': 4.7986304907386588, 'isRobot':False},\
            {'lane': -0.1256044714880388, 'pos': 5.2102349853652896, 'isRobot':False},\
            {'lane': -0.0017403147363880666, 'pos': 5.0194332605150267,'isRobot': False},\
            {'lane': -0.0017840323712066659, 'pos': 5.5120519126069265, 'isRobot':False}]

    for car_dict in car_set:
        if car_dict['isRobot']:
            world.cars.append(car.SwitchOptimizerCar(dyn, [car_dict['lane'], car_dict['pos'], math.pi/2., 0.5], color='red', iamrobot=True))
        else:
            world.cars.append(car.SimpleOptimizerCar(dyn, [car_dict['lane'], car_dict['pos'], math.pi/2., 0.5], color='yellow'))
       

    return world    

def world10():

    dyn = dynamics.CarDynamics(0.1)
    world = World(midlevel_exists=True)
    clane = lane.StraightLane([0., -1.], [0., 1.], lane.DEFAULT_WIDTH)
    world.lanes += [clane, clane.shifted(1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-1), clane.shifted(2.5), clane.shifted(-1.5)]
    world.cars.append(car.SwitchOptimizerCar(dyn, [-0.13, 0., math.pi/2., 0.5], color='red', iamrobot = True))
    world.cars.append(car.SwitchOptimizerCar(dyn, [-0.13, 0.3, math.pi/2., 0.5], color='red', iamrobot=True))
    world.cars.append(car.SwitchOptimizerCar(dyn, [-0.13, -0.3, math.pi/2., 0.5], color='red', iamrobot = True))
    world.cars.append(car.SwitchOptimizerCar(dyn, [0, -0.2, math.pi/2., 0.5], color='red', iamrobot = True))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0, 0.15, math.pi/2., 0.5], color='yellow', iamrobot = False))
    world.cars.append(car.SwitchOptimizerCar(dyn, [0, 0.4, math.pi/2., 0.5], color='red', iamrobot = True))


    world.cars[0].baseline_reward = world.simple_reward(world.cars[0], speed=0.6)
    world.cars[0].default_u = np.asarray([0., 1.])
    world.cars[1].baseline_reward = world.simple_reward(world.cars[1], speed=0.6) 
    world.cars[1].default_u = np.asarray([0., 1.])
    world.cars[2].baseline_reward = world.simple_reward(world.cars[2], speed=0.6) 
    world.cars[2].default_u = np.asarray([0., 1.])
    world.cars[3].baseline_reward = world.simple_reward(world.cars[3], speed=0.6) 
    world.cars[3].default_u = np.asarray([0., 1.])    
    world.cars[4].reward = world.simple_reward(world.cars[4], speed=0.6) 
    world.cars[4].default_u = np.asarray([0., 1.])
    world.cars[5].baseline_reward = world.simple_reward(world.cars[5], speed=0.6) 
    world.cars[5].default_u = np.asarray([0., 1.])        

    return world        

def world11():

    dyn = dynamics.CarDynamics(0.1)
    world = World(midlevel_exists=True)
    clane = lane.StraightLane([0., -1.], [0., 1.], lane.DEFAULT_WIDTH)
    world.lanes += [clane, clane.shifted(1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-1), clane.shifted(2.5), clane.shifted(-1.5)]
    world.cars.append(car.SwitchOptimizerCar(dyn, [-0.13, 0.3, math.pi/2., 0.5], color='red', iamrobot = True))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, 0., math.pi/2., 0.5], color='yellow', iamrobot=False))
    world.cars.append(car.SwitchOptimizerCar(dyn, [-0.13, -0.3, math.pi/2., 0.5], color='red', iamrobot = True))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0, 0.1, math.pi/2., 0.5], color='yellow', iamrobot = False))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0, -0.25, math.pi/2., 0.5], color='yellow', iamrobot = False))

    world.cars[0].baseline_reward = world.simple_reward(world.cars[0], speed=0.6)
    world.cars[0].default_u = np.asarray([0., 1.])
    world.cars[1].reward = world.simple_reward(world.cars[1], speed=0.6) 
    world.cars[1].default_u = np.asarray([0., 1.])
    world.cars[2].baseline_reward = world.simple_reward(world.cars[2], speed=0.6) 
    world.cars[2].default_u = np.asarray([0., 1.])
    world.cars[3].reward = world.simple_reward(world.cars[3], speed=0.6) 
    world.cars[3].default_u = np.asarray([0., 1.])    
    world.cars[4].reward = world.simple_reward(world.cars[4], speed=0.6) 
    world.cars[4].default_u = np.asarray([0., 1.])  

    return world


#TODO: define a world for testing nominal follow distance for human drivers

def world_features(num=0):
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.13)
    world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads += [clane]
    world.fences += [clane.shifted(2), clane.shifted(-2)]
    world.cars.append(car.UserControlledCar(dyn, [-0.13, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.Car(dyn, [0., 0.1, math.pi/2.+math.pi/5, 0.], color='yellow'))
    world.cars.append(car.Car(dyn, [-0.13, 0.2, math.pi/2.-math.pi/5, 0.], color='yellow'))
    world.cars.append(car.Car(dyn, [0.13, -0.2, math.pi/2., 0.], color='yellow'))
    #world.cars.append(car.NestedOptimizerCar(dyn, [0.0, 0.5, math.pi/2., 0.3], color='yellow'))
    return world

if __name__ == '__main__':
    world = playground()
    #world.cars = world.cars[:0]
    vis = visualize.Visualizer(0.1, magnify=1.2)
    vis.main_car = None
    vis.use_world(world)
    vis.paused = True
    @feature.feature
    def zero(t, x, u):
        return 0.
    r = zero
    #for lane in world.lanes:
    #    r = r+lane.gaussian()
    #for fence in world.fences:
    #    r = r-3.*fence.gaussian()
    r = r - world.cars[0].linear.gaussian()
    #vis.visible_cars = [world.cars[0]]
    vis.set_heat(r)
    vis.run()
