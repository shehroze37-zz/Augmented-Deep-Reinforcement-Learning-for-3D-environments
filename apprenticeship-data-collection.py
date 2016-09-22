#!/usr/bin/env python

from __future__ import print_function
from time import sleep
from vizdoom import *


import cv2 # for heatmap debugging
import numpy as np
import sys
import os
import ast
import cheat

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

# game.load_config("../../examples/config/basic.cfg")
game.load_config("../../examples/config/deadly_corridor.cfg")
#game.load_config("../../examples/config/deathmatch.cfg")
# game.load_config("../../examples/config/defend_the_center.cfg")
# game.load_config("../../examples/config/defend_the_line.cfg")
# game.load_config("../../examples/config/health_gathering.cfg")
# game.load_config("../../examples/config/my_way_home.cfg")
# game.load_config("../../examples/config/predict_position.cfg")
#game.load_config("../../examples/config/take_cover.cfg")

# Enables freelook in engine
game.add_game_args("+freelook 1")

game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_doom_map("map01")  # Full deathmatch.

# Enables spectator mode, so you can play. Sounds strange but it is agent who is supposed to watch not you.
game.set_window_visible(True)
game.set_render_weapon(True)
game.set_mode(Mode.SPECTATOR)
game.set_render_hud(True)

human_actions = []
total_action_count = 0

action_dictionary = {}

game.init()

total_actions = 0

'''
available_buttons = 
	{ 
		ATTACK 
		SPEED        	 
		MOVE_RIGHT 
		MOVE_LEFT 
		MOVE_BACKWARD
		MOVE_FORWARD
		TURN_RIGHT 
		TURN_LEFT 
		SELECT_WEAPON5 
	}
'''


mapping = {
        str([0,0,0,0,0,0,0,0,0]) : 1,
        str([1,0,0,0,0,0,0,0,0]) : 2,
        str([0,0,0,0,1,0,0,0,0]) : 3,
        str([0,0,0,0,0,1,0,0,0]) : 4,
        str([0,0,0,0,0,0,1,0,0]) : 5,
        str([0,0,0,0,0,0,0,1,0]) : 6,
        str([0,0,0,0,0,0,0,0,1]) : 7,
        str([0,0,0,0,0,1,0,1,0]) : 8,
        str([1,0,0,0,0,1,0,0,0]) : 9,
        str([0,0,0,0,0,1,1,0,0]) : 10,
        str([1,0,0,0,1,0,0,0,0]) : 11,
        str([1,0,0,0,0,0,1,0,0]) : 12,
        str([1,0,0,0,0,0,0,1,0]) : 13,
        str([1,0,1,0,0,0,0,0,0]) : 14,
        str([1,0,0,0,1,0,1,0,0]) : 15,
        str([1,0,0,0,0,1,0,1,0]) : 16,
        str([0,0,0,1,0,1,0,0,0]) : 17,
        str([0,0,0,0,1,0,1,0,0]) : 18,
        str([1,0,0,0,1,0,0,0,0]) : 19,
	str([1,0,0,0,0,1,1,0,0]) : 20
}

def actionAllowed(a):

    if str(a) in mapping.keys():
        return True
    else:
        return False

def mapAction(a):
    return mapping[str(a)]    

def getStringForAction(a):

    available_buttons = {0 : 'ATTACK', 1 : 'SPEED' , 2 : 'MOVE_RIGHT' ,3 : 'MOVE_LEFT', 4 : 'MOVE_BACKWARD',5 : 'MOVE_FORWARD',6 : 'TURN_RIGHT',7 : 'TURN_LEFT', 8 : 'SELECT_WEAPON5' }

    action_string = ''
    a = ast.literal_eval(a)

    for i in range(len(a)):
	if a[i] == 1:
		action_string += available_buttons[i] + ' , '

    return action_string
				

def convert(img):
    img = img[0].astype(np.float32) 
    img = cv2.resize(img, (84, 84))
    img = img.reshape([84*84])
    return img

def getHeatMapForStoring(heatmap):
    heatmap = cv2.resize(heatmap, )

episodes = 300

total_frames = 0


for i in range(128,129):
    
    print("Episode #" + str(i + 1))

    current_global_count = 0
    initial_capacity = 10
    all_states = np.zeros((initial_capacity, 4, 120 * 120), dtype=np.float32)
    all_actions = np.zeros((initial_capacity, 1), dtype=np.int32)
    all_images  = np.zeros((initial_capacity, 4, 84*84), dtype=np.float32)


    current_recent_count = 0
    state_shape = (4, 120 * 120)
    recent_states = np.zeros(state_shape, dtype=np.float32)
    recent_state_image = np.zeros((4, 84*84), dtype=np.float32)
    recent_action = np.zeros((4), dtype=np.int32)

    game.new_episode()
    tick = 0
    total_game_variable_kills = 0
    total_game_kills = 0

    rewards = []

    complete_heatmap_list = []
    complete_action_list = []

    main_directory = 'apprenticeship-data/' + str(i) + '/'

    if not os.path.isdir(main_directory):
	   os.makedirs(main_directory)


    while not game.is_episode_finished():

        tick += 1

        s = game.get_state()
        img = s.image_buffer

        misc = s.game_variables

    	heatmap = game.get_heat_maps()
	walls   = heatmap[0]
	player  = 0.8 * heatmap[1]
	medkits = 0.6 * heatmap[2]
	ammo    = 0.4 * heatmap[3]
	enemies = 0.2 * heatmap[4]
	
	#cheat.info_thing_print(game)
	print('---------------------------------------')

	goal = 1
    	if goal == 3:
		# map with enemies 
		net_in = heatmap[0] + 0.7 * heatmap[1] + 0.4 * heatmap[4] 
	elif goal == 2:
		# map with all ammo
		net_in = heatmap[0] + 0.85 * heatmap[1] + 0.7 * heatmap[3] +  0.55 * heatmap[4] + 0.4 * heatmap[8] + 0.25 * heatmap[10]
	elif goal == 1:
		# map with all health kits
		net_in = heatmap[0] + 0.8 * heatmap[1] + 0.6 * heatmap[7] 
	else:
		print('ERROR')
		sys.exit()

    	net_in = net_in / 255.0
    	'''cv2.imwrite('heatmap_' + str(tick) + '.png', net_in )'''

    	#cv2.imshow('heatmap', net_in)
        #cv2.waitKey(1)

        game.advance_action()
        a = game.get_last_action()

	print(a)

        if current_recent_count < 3:
	    if actionAllowed(a):
		    action_mapping = mapAction(a)
		    if action_mapping == 0:
			print(a)
			sys.exit()
		    recent_states[current_recent_count] = net_in.reshape([120 * 120])
		    recent_state_image[current_recent_count] = convert(img)
		    recent_action[current_recent_count] = action_mapping
		    current_recent_count = current_recent_count + 1
        else:

            if actionAllowed(a):
                '''add action and update global state and action matrix'''
                recent_states[current_recent_count] = net_in.reshape([120 * 120])
		action_mapping = mapAction(a)
		recent_state_image[current_recent_count] = convert(img)
	
		if action_mapping == 0:
			print(a)
			sys.exit()		

                recent_action[current_recent_count] = action_mapping

                if current_global_count == initial_capacity:
                    all_states.resize((initial_capacity + 1, 4, 120 * 120))
		    all_images.resize((initial_capacity + 1, 4, 84 * 84))
                    all_actions.resize((initial_capacity + 1, 1))
                    initial_capacity = initial_capacity + 1
                
                all_states[current_global_count]  = recent_states
                all_actions[current_global_count] = action_mapping
		all_images[current_global_count]  = recent_state_image

		total_frames += 4
                current_global_count = current_global_count + 1

                recent_states = np.zeros((4, 120 * 120), dtype=np.float32)
                recent_action = np.zeros((4), dtype=np.int32)
		recent_state_image = np.zeros((4, 84 * 84), dtype=np.float32)

            	current_recent_count = 0


        r = game.get_last_reward()

    	misc = s.game_variables
    	total_game_variable_kills = misc[1]

    	if a not in human_actions:
    		human_actions.append(a)
    		action_dictionary[str(a)] = 1
    		total_actions += 1
    	else:
    		action_dictionary[str(a)] = action_dictionary[str(a)] + 1
    		total_actions += 1

    	if r < -0.01:
    		break
    	if r > 0.01:
    		total_game_kills += 1
    		rewards.append(r)
    		print('Action reward = ' + str(r))

        
	#print('Weapon = ' + str(game.get_state().game_variables[3]))

        '''cv2.imshow('heatmap', heatmap[0])
        cv2.waitKey(1)'''

        '''print("state #" + str(s.number))
        print("game variables: ", misc)'''
        '''print("action:", a)
        print("reward:", r)'''
        '''print("=====================")'''

    print("episode finished!")

    print("Total states stored = " + str(current_global_count))
    print("Total actions stored = " + str(current_global_count))
    print('Game var killed = ' + str(total_game_variable_kills))
    print('Total game kills = ' + str(total_game_kills))
    print("total reward:", game.get_total_reward())
    print("Total actions = " + str(len(human_actions)))

    print("Saving training data with total frames = " + str(total_frames))
    net_input_file = open(main_directory + 'network_input_heatmap_120', 'w')
    image_state_file = open(main_directory + 'network_image_84', 'w')
    action_file = open(main_directory + 'actions', 'w')

    np.save(net_input_file, all_states)	    
    np.save(action_file, all_actions)
    np.save(image_state_file, all_images)

    net_input_file.close()
    action_file.close()
    image_state_file.close()

    print("************************")
    sleep(2.0)
   
   
'''for key,value in action_dictionary.iteritems():
	percentage = (value / float(total_actions)) * 100
	print('Percentage for action ' + getStringForAction(key) + ' = ' + str(percentage) + '')''' 

game.close()
