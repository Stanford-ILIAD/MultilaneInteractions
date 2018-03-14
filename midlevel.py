import lane
import car
import feature
import theano as th
import theano.tensor as tt

from copy import deepcopy


import math
import numpy as np

#in meters
VEH_LEN = 4.

"""
Mid-level optimizer

four phases:
0. Cars intermixed
1. Autonomous vehicles following optimal routing
2. human vehicles following optimal routing
3. autonomous vehicles fully platooned in mixed lane
"""

class Midlevel(object):
	def __init__(self, world, lanes, cars):

		# TODO: this is temporary, don't need to store the whole world, just simple_reward
		#self.world = world
		self.fences = world.fences
		self.roads = world.roads

		# reorder lanes to make sure it is in order of increasing center x-coordinate
		self.lanes = sorted(lanes, key=lambda lane: lane.p[0])
		#self.lanes = lanes
		self.cars = cars
		self.num_lanes = len(lanes)
		self.cur_phase = 0
		self.lane_centers = []		# the position of the center of each lane
		for l in self.lanes:
			self.lane_centers.append(l.p[0])
		self.veh_lanes = {}			# which lane each vehicle is in
		self.vehicle_desired_lane = []	# where each vehicle should end up
		# Find the number of human and robot cars on the road
		self.num_hum_cars = 0
		self.num_rob_cars = 0
		self.hum_cars = set()	# set of all human vehicles
		self.rob_cars = set()	# set of all robot vehicles
		idx = 0
		for c in self.cars:
			if c.iamrobot:
				self.num_rob_cars += 1
				self.rob_cars.add(c)
				idx += idx
			else:
				self.num_hum_cars += 1
				self.hum_cars.add(c)
				idx += idx
		# keep track of optimal routing:
		self.opt_rob_routing = [0]*self.num_lanes
		self.opt_hum_routing = [0]*self.num_lanes
		self.mixed_lane_idx = None
		self.n_rob_lanes = 0
		self.n_hum_lanes = 0
		self.rob_cars_lane_asg = {}

		# keep track of how many times goals have been assigned in each phase
		self.phase_0_asgn_counter = 0
		self.phase_1_asgn_counter = 0
		self.phase_2_asgn_counter = 0

		# find our initial state
		self.find_veh_lanes()


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


	# determine which lane each vehicle is in.
	# we do this by finding the lane center closest to the vehicle's x-position
	def find_veh_lanes(self):

		# find which lane each car is in
		for c in self.cars:
			self.veh_lanes[c] = min(range(len(self.lane_centers)), key=lambda x:abs( self.lane_centers[x] - c.x[0]) )

		# sanity check
		assert(len(self.veh_lanes) == len(self.cars))

	# The following function should return with how many regular
	# and how many smart vehicles should be in each lane
	# TODO: implement this!
	def find_optimal_routing(self):

		# find how many lanes should be purely robot and how many purely human
		#TODO get these parameters from somewhere else
		plat_dist = 4.
		#non_plat_dist = 8.
		# EXTREMELY TEMPORARY:
		non_plat_dist = 12.
		veh_len = VEH_LEN
		aut_level = float(self.num_rob_cars) / float(self.num_rob_cars + self.num_hum_cars)
		k1 = non_plat_dist + veh_len
		k2 = non_plat_dist - plat_dist
		m = int(math.floor(aut_level * self.num_lanes * (k1 - k2)/(k1 - aut_level * k2) ) ) # number of autonomous lanes

		# We shouldn't have all autonomous lanes or negative autonomous lanes
		assert(m >= 0 and m < int(self.num_lanes) )
		# assert(m>0) #just for now, so we can avoid the difficult situation of when there are no purely autonomous lanes

		# Note that if we are going all the way to phase 3, routing should be optimal with respect to the upper bound capacity function
		"""

		# Now find the autonomy level in the mixed lane
		s = -float(m) * (k1 - aut_level * k2) + (self.num_lanes-1.) * aut_level * (k1-k2)
		p = [0,0,0]
		p[0] = k2 * s
		p[1] = k1*(k1-k2)
		p[2] = -k1 * (aut_level * (k1-k2) + s)

		roots = np.roots(p)
		assert(len(roots) == 2)
		if (roots[0] >= 0 and roots[0] <= 1) :
			mixed_aut_lev = roots[0]
		elif (roots[1] >= 0 and roots[1] <= 1):
			mixed_aut_lev = roots[1]
		else:
			assert(1==2) # this should not happen

		def cap(aut_lev, c1, c2) :
			return 1./(c1 - (aut_lev ** 2) * c2)
		"""

		s = float(m)*(1.-aut_level)/(k1-k2) - (self.num_lanes-m-1)*aut_level/k1
		mixed_aut_lev = (aut_level-k1*s)/(1-k2*s)

		def capUB(aut_lev, c1, c2) :
			return 1./(c1 - aut_lev * c2)

		# Now translate this to a number of vehicles in each lane
		tot_rob_cap = float(m) * capUB(1., k1, k2) + mixed_aut_lev * capUB(mixed_aut_lev, k1, k2)
		tot_hum_cap = (self.num_lanes-m-1) * capUB(0., k1, k2) + (1. - mixed_aut_lev) * capUB(mixed_aut_lev, k1, k2)
		robs_per_rob_lane = int(math.floor(self.num_rob_cars * capUB(1., k1, k2) / tot_rob_cap ) )
		robs_per_mixed_lane = int( math.floor(self.num_rob_cars * mixed_aut_lev * capUB(mixed_aut_lev, k1, k2) / tot_rob_cap ) )
		hums_per_hum_lane = int( math.floor(self.num_hum_cars * capUB(0., k1, k2) / tot_hum_cap ) )
		hums_per_mixed_lane = int( math.floor(self.num_hum_cars * (1.-mixed_aut_lev) * 
			capUB(mixed_aut_lev, k1, k2) / tot_hum_cap ) )

		allocated_robs = float(m)*robs_per_rob_lane + robs_per_mixed_lane
		allocated_hums = (self.num_lanes-m-1)*hums_per_hum_lane + hums_per_mixed_lane

		assert( allocated_robs <= self.num_rob_cars )
		assert( allocated_hums <= self.num_hum_cars )

		# now fill up our assignment vectors, adding in cars that got lost when we rounded down
		
		for i in range(0, self.num_lanes):
			if i < m:
				self.opt_hum_routing[i] = 0
				self.opt_rob_routing[i] = robs_per_rob_lane
				if allocated_robs < self.num_rob_cars:
					self.opt_rob_routing[i] +=  1
					allocated_robs += 1
			elif i == m:
				self.opt_rob_routing[i] = robs_per_mixed_lane
				self.opt_hum_routing[i] = hums_per_mixed_lane
			else:
				self.opt_hum_routing[i] = hums_per_hum_lane
				self.opt_rob_routing[i] = 0
				if allocated_hums < self.num_hum_cars:
					self.opt_hum_routing[i] += 1
					allocated_hums += 1

		# if not all cars are assigned, add to the mixed lane
		# because of zero-indexing, m is the index of the mixed lane
		if allocated_robs < self.num_rob_cars:
			self.opt_rob_routing[m] += 1
			allocated_robs += 1
		if allocated_hums < self.num_hum_cars:
			self.opt_hum_routing[m] += 1
			allocated_hums += 1

		# at this point all vehicles should be allocated
		assert(allocated_robs == self.num_rob_cars)
		assert(allocated_hums == self.num_hum_cars)

		# find number of robot-only and human-only lanes
		# find mixed lane index
		# also make sure that we have at most one mixed lane
		num_mixed_lanes = 0
		for i in range(self.num_lanes):
			if( (self.opt_rob_routing[i] != 0) and (self.opt_hum_routing[i]!= 0) ):
				num_mixed_lanes += 1
		assert(num_mixed_lanes < 2)

		# find the mixed lane, if it exists
		# also find the number of robot-only and human-only lanes
		self.mixed_lane_idx = m 
		self.n_rob_lanes = m
		self.n_hum_lanes = self.num_lanes - m - 1

		print('number of human drivers:')
		print(self.num_hum_cars)
		print('num allocated humans:')
		print(allocated_hums)

		print('optimal robot routing:')
		print(self.opt_rob_routing)
		print('optimal human routing:')
		print(self.opt_hum_routing)

	# Once we have optimal routing, find vehicles that are where they should be
	# and assign them to that lane
	def assign_optimal_routing(self):

		# first update our knowledge of where the vehicles are
		self.find_veh_lanes()

		# then create a list of how many cars have yet to be assigned to a lane, then update
		self.rob_cars_needed_in_lanes = list(self.opt_rob_routing)

		# we are only concerned with where the autonomous vehicles are for now (not humans)

		# use this variable to make sure our vehicle has not been assigned yet
		veh_assigned = {}

		# First find vehicles that are where they should be and assign them to that lane
		# Then find vehicles that are in the lane adjacent to a desired lane, and assign them that lane.
		# and so on. This way vehicles will be assigned lanes closest to them
		for c in self.cars:
			if(c.iamrobot):			
				l = self.veh_lanes[c]	# robot car's lane
				if(self.rob_cars_needed_in_lanes[l] > 0):
					self.rob_cars_lane_asg[c] = l
					self.rob_cars_needed_in_lanes[l] -= 1
					veh_assigned[c] = True

		for shift in range(self.num_lanes):

			for c in self.rob_cars:
				
				# make sure the car hasn't been assigned yet
				if c not in veh_assigned:

					l = self.veh_lanes[c]	# robot car's lane
					
					# check if the car can be places 'shift' lanes to the left or right
					if(l>=shift):
						if(self.rob_cars_needed_in_lanes[l - shift] > 0):
							self.rob_cars_lane_asg[c] = l - shift
							self.rob_cars_needed_in_lanes[l - shift] -= 1
							veh_assigned[c] = True
						elif(l<self.num_lanes-shift):
							if(self.rob_cars_needed_in_lanes[l + shift] > 0):
								self.rob_cars_lane_asg[c] = l + shift
								self.rob_cars_needed_in_lanes[l + shift] -= 1
								veh_assigned[c] = True
					elif(l<self.num_lanes-shift):
							if(self.rob_cars_needed_in_lanes[l + shift] > 0):
								self.rob_cars_lane_asg[c] = l + shift
								self.rob_cars_needed_in_lanes[l + shift] -= 1
								veh_assigned[c] = True

		# By now every vehicle should be assigned a lane. Check that this is the case
		for c in self.rob_cars:
			assert(veh_assigned[c] == True)

	# Check if we have completed phase 0. No return value, just updates the current phase accordingly
	def check_phase_zero(self):
		#only do this if we are in phase 0
		if (self.cur_phase != 0):
			return

		# first update our knowledge of where the vehicles are
		self.find_veh_lanes()

		# check how many robot vehicles are in each lane
		rob_vehs_in_lane = [0] * self.num_lanes
		for c in self.rob_cars:
			rob_vehs_in_lane[self.veh_lanes[c]] += 1

		optimal_rob_routing = True
		for i in range(self.num_lanes):
			if(rob_vehs_in_lane[i] != self.opt_rob_routing[i]):
				optimal_rob_routing = False

		if(optimal_rob_routing):
			self.cur_phase = 1
			return

	# Check if we have completed phase 1. No return value, just updates the current phase accordingly
	def check_phase_one(self):
		if(self.cur_phase !=1):
			return
		# first update our knowledge of where the vehicles are
		self.find_veh_lanes()

		# Check how many vehicles of each type are in each lane
		rob_vehs_in_lane = [0] * self.num_lanes
		hum_vehs_in_lane = [0] * self.num_lanes
		for c in rob_cars:
			rob_vehs_in_lane[self.veh_lanes[c]] += 1
		for c in hum_cars:
			hum_vehs_in_lane[self.veh_lanes[c]] += 1

		# make sure all robots are in proper lanes and that there 
		# are no human cars in the robot lanes
		robots_alone = True
		for i in range(self.num_lanes):
			if(rob_vehs_in_lane[i] != self.opt_rob_routing[i]):
				robots_alone = False
			if( (self.opt_hum_routing[i] == 0) and (hum_vehs_in_lane[i]!= 0) ):
				robots_alone = False

		if(robots_alone):
			self.cur_phase = 2

	# Check if we have completed phase 2
	def check_phase_two(self):

		# first update our knowledge of where the vehicles are
		self.find_veh_lanes()

		# if there is no mixed lane then we have completed phase 2
		if(self.mixed_lane_idx is None):
			self.cur_phase = 3
			return

		# collect the indices of all cars in the mixed lane then order them by position
		mixed_lane_car_idcs = []
		for i in range( len(self.cars) ):
			if(self.veh_lanes[self.cars[i]] == self.mixed_lane_idx):
				mixed_lane_car_idcs.append(i)

		# create a list of the vehicle y-positions in the mixed lane
		mixed_veh_ypos = [ self.cars[i].x[1] for i in mixed_lane_car_idcs ]

		# now sort these indices according to vehicle y-position
		sorted_idcs = np.argsort(mixed_veh_ypos)

		# sanity check
		assert( len(sorted_idcs) == len(mixed_lane_car_idcs) )

		# now check that we have the desired number of smart cars in the mixed lane
		num_smart = [self.cars[i].iamrobot for i in mixed_lane_car_idcs].count(True)

		# if we don't have the required number, we are still in phase 2, mid-maneuver
		if num_smart != self.opt_rob_routing[self.mixed_lane_idx]:
			return

		optimal_vehicle_placement = True
		switched_to_human = False	# update this to true when we go from robot to human. if we then go
									# back to robot again then we are not at optimal vehicle placement
		prev_rob = None				# previous iterated car is a robot
		for i in sorted_idcs :
			# mixed_lane_car_idcs[sorted_idcs[i]] will iterate through our mixed cars in order of position

			if(i==sorted_idcs[0]):
				prev_rob = self.cars[ mixed_lane_car_idcs[ i ] ].iamrobot
			# if we switch from robot var to human car
			elif(prev_rob and not self.cars[ mixed_lane_car_idcs[ i ] ].iamrobot):
				switched_to_human = True
				prev_rob = False
			# if we switch from human to robot car
			elif(not prev_rob and self.cars[ mixed_lane_car_idcs[ i ] ].iamrobot):
				if(switched_to_human):
					# then we are not optimal
					optimal_vehicle_placement = False
				else:
					prev_rob = True

		if(optimal_vehicle_placement):
			self.cur_phase = 3

	# Assign robot rewards to get them into their optimal lanes
	# TODO: how do we make sure this doesn't get called when a vehicle is in the middle of a maneuver?
	def assign_goals_phase_zero(self):

		# make sure we are in the correct phase
		assert(self.cur_phase == 0)

		# first update our knowledge of where the vehicles are
		self.find_veh_lanes()


		for c in self.rob_cars:
			if(self.veh_lanes[c] == self.rob_cars_lane_asg[c]):
				# change the a simple optimizer and remove the car from the list
				c.nested = False
				# put back to baseline reward
				c.simple_reward = c.baseline_reward
				

		# Now with the remaining vehicles, assign them reward functions to get closer to their goals
		# TODO: if there is a smart vehicle nearby, just go in front of that one and turn down collision avoidance for smart vehicles

		def move_to_lane(desired_xloc, width=0.5, lane_width = lane.DEFAULT_WIDTH):
			@feature.feature
			def f(t, x, u):
				return tt.exp(-0.5*((x[0]-desired_xloc)**2)/(width**2*lane_width*lane_width/4.))
			return f

		for c in self.rob_cars:
			# if the car is in the right place, make sure it goes back to its baseline reward
			if(self.veh_lanes[c] == self.rob_cars_lane_asg[c]):
				c.nested = False
				# put back to baseline reward
				c.simple_reward = c.baseline_reward
			else:
				if(self.rob_cars_lane_asg[c] > self.veh_lanes[c]):
					# Then we want to move the car to the right
					# TODO: does this actually change the reward or just in a shallow copy?
					#new_reward = c.simple_reward + move_right_reward
					new_reward = c.baseline_reward + 100.*move_to_lane(desired_xloc=self.lane_centers[self.veh_lanes[c]+1])
					move_direction = 1
				else:
					new_reward = c.baseline_reward + 100.*move_to_lane(desired_xloc=self.lane_centers[self.veh_lanes[c]-1])
					move_direction = -1

				# Find if there is a vehicle blocking -- just pair with the first vehicle behind it in the lane over
				#TODO: change this so it can be one car-length in front and still be considered blocking
				candidate = None
				for cb in self.cars:
					if cb.iamrobot:
						continue
					if self.veh_lanes[cb] == self.veh_lanes[c] + move_direction:
						if (candidate is None) and (cb.x[1] <= c.x[1]) :
							candidate = cb
						else:
							# Find which is more blocking -- closer in y-position but still behind
							if ( (cb.x[1] <= c.x[1]) and (cb.x[1] > candidate.x[1]) ):
								candidate = cb

				if candidate is None:
					c.nested = False
					c.simple_reward = new_reward
				else:
					c.nested = True
					c.human = candidate

					# create a list of all the trajectories that the human is doing collision avoidance with.
					# use true (not linear) trajectory for the robot car, since it is a Stackelberg game.

					# TODO: only use the cars close to the human
					trajs_to_avoid = []
					for ca in self.cars:
						if ca is c:
							trajs_to_avoid.append(ca.traj)
						elif ca is not candidate:
							trajs_to_avoid.append(ca.linear)

					#TODO: add speed into simple reward! 
					hum_reward = self.simple_reward(trajs_to_avoid) + 100.*feature.bounded_control(candidate.bounds)

					c.nested_rewards = (hum_reward, new_reward) # This assumes perfect knowledge of human reward functions


	# Assign robot goals to push human drivers out of the purely robot lanes
	def assign_goals_phase_one(self):

		assert(self.cur_phase == 1)

		# first update our knowledge of where the vehicles are
		self.find_veh_lanes()

		# start with left-most lane then work rightwards. Stop when we find a lane that is not done or we reach the mixed lane
		for i in range(int(self.mixed_lane_idx)):
			# check if the lane is human-free and if the cars are platooned
			lane_completed = True

			for c in self.cars:
				if (self.veh_lanes[c] == i):
					#if ( (not c.iamrobot) or (c.platoon_behind is None) ):
					if not c.iamrobot:
						lane_completed = False

			if not lane_completed:
				# temporary: only assign goals if they haven't been assigned for that lane yet
				if self.phase_1_asgn_counter <= i:
					self.move_humans_from_lane(i)
					self.phase_1_asgn_counter += 1
				return

		# If we reach here that means all robot lanes are organized, and phase 1 is over
		# TODO: remove redundant function check_phase_one
		self.cur_phase = 2


	def move_humans_from_lane(self, lane_idx):


		def move_from_lane(rob_car, desired_xloc, width=0.5, lane_width = lane.DEFAULT_WIDTH):
			@feature.feature
			def f(t, x, u):
				return tt.exp(-0.5*((rob_car.traj_h.x[t][0]-desired_xloc)**2)/(width**2*lane_width*lane_width/4.))
			return f

		# collect the cars in the lane of interest
		lane_car_idcs = []
		for i in range(len(self.cars)):
			if(self.veh_lanes[self.cars[i]] == lane_idx):
				lane_car_idcs.append(i)

		# Order the cars by position

		# create a list of the vehicle y-positions in the mixed lane
		mixed_veh_ypos = [ self.cars[i].x[1] for i in lane_car_idcs ]

		# now sort these indices according to vehicle y-position
		sorted_idcs = np.argsort(np.array(mixed_veh_ypos))

		# Now go through these cars and see if there is a human vehicle behind them.
		# If so, get the human to leave the lane. If not, if there is a robot in front,
		# platoon with them
		for i in range(len(sorted_idcs)):

			acting_car = self.cars[lane_car_idcs[sorted_idcs[i]]]

			# only assign goals if the car is a robot
			if acting_car.iamrobot:
				# If this isn't the last car in the lane, check behind it
				if (i != 0):

					if not self.cars[lane_car_idcs[sorted_idcs[i-1]]].iamrobot:
						# then kick the car out
						acting_car.nested = True
						acting_car.human = self.cars[lane_car_idcs[sorted_idcs[i-1]]]

						rob_reward = acting_car.baseline_reward + 400.*move_from_lane(acting_car, desired_xloc=self.lane_centers[self.veh_lanes[acting_car]+1])

						#TODO: add speed into here! maybe assume that the current speed is the desired speed?? 

						# create a list of all the trajectories that the human is doing collision avoidance with.
						# use true (not linear) trajectory for the robot car, since it is a Stackelberg game.

						# TODO: only use the cars close to the human
						trajs_to_avoid = []
						for ca in self.cars:
							if ca is acting_car:
								trajs_to_avoid.append(ca.traj)
							elif ca is not self.cars[lane_car_idcs[sorted_idcs[i-1]]]:
								trajs_to_avoid.append(ca.linear)

						hum_reward = self.simple_reward(trajs_to_avoid) + 100.*feature.bounded_control(acting_car.human.bounds)

						acting_car.nested_rewards = (hum_reward, rob_reward)

					# if the car in front it is a robot, then be free to platoon (since the car behind it is a robot too)
					# check that it is not the front-most car in the lane
					elif i < len(mixed_veh_ypos)-1:
						# if there is a robot in front of it, platoon with them
						if self.cars[lane_car_idcs[sorted_idcs[i+1]]].iamrobot:
							# Then platoon with the car in front
							acting_car.nested = False
							acting_car.platoon(self.cars[lane_car_idcs[sorted_idcs[i+1]]])
				# if it is the rear-most vehicle in a lane, check that it is not the only vehicle in the lane
				elif i < len(mixed_veh_ypos)-1:
					if self.cars[lane_car_idcs[sorted_idcs[i+1]]].iamrobot:
						# Then platoon with the car in front
						acting_car.nested = False
						acting_car.platoon(self.cars[lane_car_idcs[sorted_idcs[i+1]]])
		

	# Assign robot goals to platoon the mixed lane
	# if in the situation where there is a mixed lane, we use phase_2_asgn_counter to keep
	# track of how many smart cars are platooned in the rear platoon of the mixed lane.
	def assign_goals_phase_two(self):

		m = self.mixed_lane_idx

		assert(self.cur_phase == 2)

		# first update our knowledge of where the vehicles are
		self.find_veh_lanes()	

		# collect the cars in the lane of interest
		lane_car_idcs = []
		for i in range(len(self.cars)):
			if (self.veh_lanes[self.cars[i]] == m):
				lane_car_idcs.append(i)

		# Order the cars by position. to do so, first create a list of the vehicle y-positions in the mixed lane
		mixed_veh_ypos = [ self.cars[i].x[1] for i in lane_car_idcs ]

		# now sort these indices according to vehicle y-position. this sorts into increasing order, so the first vehicle will be the rearmost vehicle
		sorted_idcs = np.argsort(np.array(mixed_veh_ypos) )

		# reorder the indices according to the y-position
		sorted_lane_car_idcs = [ lane_car_idcs[i] for i in sorted_idcs]

		# find the number of smart cars in the mixed lane:
		num_smart = [self.cars[i].iamrobot for i in sorted_lane_car_idcs].count(True)

		# for now, make sure that we have at least one smart car
		assert(num_smart > 0)

		# now find the indices of the rear platoon
		rear_platoon_idcs = []
		for i in sorted_lane_car_idcs:
			if self.cars[i].iamrobot:
				rear_platoon_idcs.insert(0, i)
			else:
				# if there is a human but we already found robots (list is not empty), return the list
				if rear_platoon_idcs:
					break


		#TODO: combine this with the other move_to_lane above
		def move_to_lane(desired_xloc, width=0.5, lane_width = lane.DEFAULT_WIDTH):
			@feature.feature
			def f(t, x, u):
				return tt.exp(-0.5*((x[0]-desired_xloc)**2)/(width**2*lane_width*lane_width/4.))
			return f

		# Note that leader_active tells us which car is maneuvering, so we know which should have true trajectory and which should be linear
		# TODO: not sure we are using the trajectories correctly here
		def platoon_car(leader, follower, leader_active, platoon_distance=0.3, width=0.5):
			@feature.feature
			def f(t, x, u):

				if leader_active:

					#distance = np.sqrt((leader.x[0]-follower.x[0])**2 + (leader.x[1]-follower.x[1])**2)
					distance = np.sqrt((leader.traj.x[t][0]-follower.linear.x[t][0])**2 + (leader.traj.x[t][1]-follower.linear.x[t][1])**2)

					# if we are not close to the goal car, it doesn't matter what lane we're in (hence the exp term -- importance of being in the correct lane decays exponentially with y-distance from target)
					#x_penalty = -10*tt.exp(-10.0*(distance-platoon_distance))*(leader.x[0]-follower.x[0])**2
					x_penalty = -10*tt.exp(-10.0*(distance-platoon_distance))*(leader.traj.x[t][0]-follower.linear.x[t][0])**2

					# The y penalty should saturate at a certain distance because a very distant car shouldn't engage in very risky maneuvers.
					# Because of this we have the exponential saturation term.
					#y_penalty = -(-1.0/2.0 + 100.0/(1.0 + tt.exp(-1.0/10.0*(leader.x[1] - follower.x[1] - platoon_distance)**2) ) )
					y_penalty = -(-1.0/2.0 + 100.0/(1.0 + tt.exp(-1.0/10.0*(leader.traj.x[t][1] - follower.linear.x[t][1] - platoon_distance)**2) ) )
				else:
					distance = np.sqrt((leader.linear.x[t][0]-follower.traj.x[t][0])**2 + (leader.linear.x[t][1]-follower.traj.x[t][1])**2)
					x_penalty = -10*tt.exp(-10.0*(distance-platoon_distance))*(leader.linear.x[t][0]-follower.traj.x[t][0])**2
					y_penalty = -(-1.0/2.0 + 100.0/(1.0 + tt.exp(-1.0/10.0*(leader.linear.x[t][1] - follower.traj.x[t][1] - platoon_distance)**2) ) )

				return x_penalty + y_penalty
			return f


		# this will be very different depending on if any lanes are purely robot car
		if m == 0:

			# only assign goals if we haven't already
			if self.phase_2_asgn_counter == 0:
				
				self.phase_2_asgn_counter = 1

				first_robot = True
				# give each robot car the goal of platooning with the robot behind it. if there is a human in between, nest with that human.
				for i in range(len(sorted_idcs)):
					acting_car = self.cars[sorted_lane_car_idcs[i]]
					# only assign goals to robot cars
					if not acting_car.iamrobot:
						continue
					# no robot behind it to platoon with
					if first_robot:
						first_robot = False
						continue
					# find the first robot car behind this vehicle. look at all indices before this one
					blocking_car = None
					# if the first car behind it is a human, consider that the blocking car
					# otherwise, have it nest with the robot it is platooning with
					if not self.cars[sorted_lane_car_idcs[i-1]].iamrobot:
						blocking_car = self.cars[sorted_lane_car_idcs[i-1]]

					for j in reversed(range(i)):
						# platoon with the first robot found
						if self.cars[sorted_lane_car_idcs[j]].iamrobot:
							passive_robot = self.cars[sorted_lane_car_idcs[j]]
							if not blocking_car:
								blocking_car = passive_robot

							new_reward = acting_car.baseline_reward + 100.*platoon_car(leader=acting_car, follower=passive_robot, leader_active=True, platoon_distance=0.3)
							self.assign_nest_goal(acting_car=acting_car, passive_car=blocking_car, acting_car_reward=new_reward)
							break


			

			"""
			# Now go through these cars and see if there is a human vehicle behind them.
			# If so, get the human to leave the lane. If not, get the car behind to platoon with them

			for i in range(len(sorted_idcs)):

				acting_car = self.cars[sorted_lane_car_idcs[i]]

				# only assign goals to robot cars
				if not acting_car.iamrobot:
					continue

				# tell each smart car to platoon with the smart car behind it. nest each smart car with the intervening regular car, if it exists
				# TODO: this will cause mayhem. do this in a nicer way.

				# If this isn't the first car in the lane, check behind it
				if i != 0:
					if not self.cars[sorted_lane_car_idcs[i-1]].iamrobot:
						# then kick the car out
						acting_car.nested = True
						acting_car.human = self.cars[sorted_lane_car_idcs[i-1]]
						acting_car.reward = move_human # TODO: implement this
					# if the car in front it is a robot, then be free to platoon (since the car behind it is a robot too)
					elif (self.cars[sorted_lane_car_idcs[i+1]].iamrobot):
						# Then platoon with the car in front
						acting_car.nested = False
						acting_car.platoon(self.cars[sorted_lane_car_idcs[i+1]])
				# if it is the last car in a lane, it does not need to check behind it
				elif self.cars[sorted_lane_car_idcs[i+1]].iamrobot:
					acting_car.nested = False
					acting_car.platoon(self.cars[sorted_lane_car_idcs[i+1]])
			"""

		else:
			# Swap with the lane next to it. Form the platoon at the end of the mixed lane

			# first check what the counter says. If the counter says 0 then we have not started yet.
			if self.phase_2_asgn_counter == 0:

				self.phase_2_asgn_counter = len(rear_platoon_idcs)

			# if all smart cars in mixed lane are platooned, we have finished phase 2
			if self.phase_2_asgn_counter == num_smart:
				self.cur_phase = 3
				return


			# Check if as many cars are platooned as expected. This means that all assigned robots have finished their maneuvers
			if self.phase_2_asgn_counter == len(rear_platoon_idcs):

				# Now assign goal to the next rear-most car to get in the purely autonomous lane
				for i in sorted_lane_car_idcs:
					if self.cars[i].iamrobot and i not in rear_platoon_idcs:

						acting_car = self.cars[i]

						# assign this car to move over, nesting with the nearest vehicle behind it in the adjacent lane
						new_reward = acting_car.baseline_reward + 100.*move_to_lane(desired_xloc=self.lane_centers[m-1])

						blocking_car = self.find_blocking_car(lane=(m-1), ypos=acting_car.x[1])

						self.assign_nest_goal(acting_car=acting_car, passive_car=blocking_car, acting_car_reward=new_reward)

						break

				# if we get here that means we didn't find any cars to assign goals to. this should not happen.
				else:
					assert(1==2)

				# Now assign that a smart car go to the mixed lane. 
				acting_car = self.find_car_to_join_rear_platoon(mixed_lane=m, platoon_veh_idcs=rear_platoon_idcs)

				#new_reward = acting_car.baseline_reward + 100.*move_to_lane(desired_xloc=self.lane_centers[m])

				# check which car we are meant to nest with. if the vehicle is in front of the platoon leader than it should nest with platoon leader
				if acting_car.x[1]>=self.cars[rear_platoon_idcs[0]].x[1]:
					passive_car = self.cars[rear_platoon_idcs[0]]
					#new_reward = acting_car.baseline_reward + 100.*platoon_car(leader=acting_car, follower=passive_car, platoon_distance=0.3)
					new_reward = acting_car.baseline_reward + 100.*platoon_car(leader=acting_car, follower=passive_car, leader_active=True, platoon_distance=0.3)
				else:
					# rear platoon car is the leader
					passive_car = self.cars[rear_platoon_idcs[-1]]
					new_reward = acting_car.baseline_reward + 100.*platoon_car(leader=passive_car, follower=acting_car, leader_active=False, platoon_distance=0.3)

				# Now give platooning goals
				self.assign_nest_goal(acting_car=acting_car, passive_car=passive_car, acting_car_reward=new_reward)

				# increment our counter so we know not to assign new goals until the platoon grows in length by 1
				self.phase_2_asgn_counter += 1


	# find the vehicle that is blocking a merge
	# Note that the vehicle return type can be human or robot
	# arguments: lane is desired lane, ypos is merger's y-position
	def find_blocking_car(self, lane, ypos):

		# Find if there is a vehicle blocking -- just pair with the first vehicle behind it in the lane over
		#TODO: change this so it can be one car-length in front and still be considered blocking
		candidate = None
		for cb in self.cars:
			if self.veh_lanes[cb] == lane:
				if (candidate is None) and (cb.x[1] <= ypos) :
					candidate = cb
				else:
					# Find which is more blocking -- closer in y-position but still behind
					if ( (cb.x[1] <= ypos) and (cb.x[1] > candidate.x[1]) ):
						candidate = cb

		return candidate

	def find_car_to_join_rear_platoon(self, mixed_lane, platoon_veh_idcs):

		aut_lane = mixed_lane-1
		leader_ypos = self.cars[platoon_veh_idcs[0]].x[1]
		rear_ypos = self.cars[platoon_veh_idcs[-1]].x[1]

		# find the car in the purely autonomous lane that is closest to, and in front of, mixed platoon leader
		candidate = None
		for c in self.cars:
			if self.veh_lanes[c] == aut_lane:
				if (candidate is None) and (c.x[1]>=leader_ypos):
					candidate = c
				else:
					if( (c.x[1] >= leader_ypos) and (c.x[1]<candidate.x[1]) ):
						candidate = c

		# if we can't find a candidate still, find the vehicle closest to, and in front of, the final car in the rear platoon
		if candidate is None:
			if self.veh_lanes[c] == aut_lane:
				if (candidate is None) and (c.x[1]>=rear_ypos):
					candidate = c
				elif( (c.x[1] >= rear_ypos) and (c.x[1]<candidate.x[1]) ):
					candidate = c

		# if we still can't find a candidate, find the smart car that's nearest behind the final car in the rear platoon
		if candidate is None:
			if self.veh_lanes[c] == aut_lane:
				if (candidate is None) and (c.x[1]<=rear_ypos):
					candidate = c
				elif( (c.x[1] <= rear_ypos) and (c.x[1]>candidate.x[1]) ):
					candidate = c

		assert(candidate is not None)

		return candidate

	def assign_nest_goal(self, acting_car, passive_car, acting_car_reward):

		if passive_car is None:
			acting_car.nested = False
			acting_car.simple_reward = acting_car_reward
		else:
			acting_car.nested = True
			acting_car.human = passive_car

			# create a list of all the trajectories that the human is doing collision avoidance with.
			# use true (not linear) trajectory for the robot car, since it is a Stackelberg game.

			# TODO: only use the cars close to the human
			trajs_to_avoid = []
			for ca in self.cars:
				if ca is acting_car:
					trajs_to_avoid.append(ca.traj)
				elif ca is not passive_car:
					trajs_to_avoid.append(ca.linear)

			hum_reward = self.simple_reward(trajs_to_avoid) + 100.*feature.bounded_control(passive_car.bounds)

			acting_car.nested_rewards = (hum_reward, acting_car_reward)


	"""
	def find_rear_platoon_length(self):

		self.find_veh_lanes()

		# collect the indices of the cars in the lane of interest, ordered by y-position
		lane_car_idcs = []
		for i in range(self.cars):
			if (self.veh_lanes[self.cars[i]] == self.mixed_lane_idx):
				lane_car_idcs.append(i)		

		mixed_veh_ypos = [ self.cars[i].x[1] for i in lane_car_idcs ]
		sorted_idcs = np.argsort(np.array(mixed_veh_ypos) )	
		sorted_lane_car_idcs = [ lane_car_idcs[i] for i in sorted_idcs]

		# now find the indices of the rear platoon. start with the rear car
		rear_platoon_idcs = []

		for i in sorted_lane_car_idcs:
			if self.cars[i].iamrobot:
				rear_platoon_idcs.insert(0, i)
			else:
				# if there is a human but we already found robots (list is not empty), return the list
				if rear_platoon_idcs:
					return rear_platoon_idcs
	"""

	
	# Call this function repeatedly to update the goals
	def increment_goals(self):

		print('We are at phase:'+str(self.cur_phase))

		# If we haven't found the optimal routing yet, find it
		if self.opt_rob_routing == [0]*self.num_lanes:
			self.find_veh_lanes()
			self.find_optimal_routing()
			self.assign_optimal_routing()

		if self.cur_phase == 0:

			self.check_phase_zero()

			# If the current phase is the same, do the action.
			# TODO: clean this up
			if self.cur_phase == 0:
				# temporary: only assign goals if they haven't been assigned yet
				if self.phase_0_asgn_counter == 0:
					self.assign_goals_phase_zero()
					self.phase_0_asgn_counter += 1
			else:
				# help speed up the simulation a little bit
				self.assign_goals_phase_one()
				if self.cur_phase == 2:
					self.check_phase_two()
					self.assign_goals_phase_two()


		# This one automatically checks if the phase is complete
		elif self.cur_phase == 1:
			self.assign_goals_phase_one()
			# speed up sim
			if self.cur_phase == 2:
				self.check_phase_two()
				self.assign_goals_phase_two()


		elif self.cur_phase == 2:

			self.check_phase_two()

			if self.cur_phase==2:
				self.assign_goals_phase_two()
			
			
		elif self.cur_phase == 3:
			placeholder = 1

	







