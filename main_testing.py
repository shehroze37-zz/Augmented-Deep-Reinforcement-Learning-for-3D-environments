import lua
from vizdoom import *
from random import choice
from time import sleep
from time import time

import itertools as it
import cv2
import numpy as np
import argparse
import sys

from opencv.cv import *
from opencv.highgui import *

torch = lua.require('torch')
lua.require('trepl')
lua.require('cunn') # We run the network on GPU
dqn = lua.eval("dofile('dqn/NeuralQLearner.lua')")
tt = lua.eval("dofile('dqn/TransitionTable.lua')")
#lua.execute("dofile('dqn/Scale.lua')") # for the preproc

spectator_action_mapping = {
        1 : [0,0,0,0,0,0,0,0,0] ,
        2 : [1,0,0,0,0,0,0,0,0] ,
        3 : [0,0,0,0,1,0,0,0,0] ,
        4 : [0,0,0,0,0,1,0,0,0] ,
        5 : [0,0,0,0,0,0,1,0,0] ,
        6 : [0,0,0,0,0,0,0,1,0] ,
        7 : [0,0,0,0,0,0,0,0,1] ,
        8 : [0,0,0,0,0,1,0,1,0] ,
        9 : [1,0,0,0,0,1,0,0,0] ,
        10 : [0,0,0,0,0,1,1,0,0] ,
        11 : [1,0,0,0,1,0,0,0,0] ,
        12 : [1,0,0,0,0,0,1,0,0] ,
        13 : [1,0,0,0,0,0,0,1,0] ,
        14 : [1,0,1,0,0,0,0,0,0] ,
        15 : [1,0,0,0,1,0,1,0,0] ,
        16 : [1,0,0,0,0,1,0,1,0] ,
        17 : [0,0,0,1,0,1,0,0,0] ,
        18 : [0,0,0,0,1,0,1,0,0] ,
        19 : [1,0,0,0,1,0,0,0,0] ,
	20 : [1,0,0,0,0,1,1,0,0] 
}

def getScreen(game, doom_env, network_input, heatmap_type,video_folder=None,video_count=0, args):

	screen, heatmap = None, None
	if network_input == 'image':
	    screen = doom_env.convert(game.get_state().image_buffer).reshape([1, 1, args.height, args.width])
	elif network_input == 'heat_map':

	    heatmap = getScreenFromHeatMap(game.get_heat_maps(), heatmap_type)
	    if args.display == 1:
		    cv2.imshow('heatmap', heatmap)
			cv2.waitKey(1)
	elif network_input == 'both':

		screen = doom_env.convert(game.get_state().image_buffer).reshape([1, 1, args.height, args.width])
		heatmap = getScreenFromHeatMap(game.get_heat_maps(), heatmap_type)
	    if args.display == 1:
		    cv2.imshow('heatmap', heatmap )
			cv2.waitKey(1)
	else:
		print('ERROR: : WRONG input type')
		sys.exit()

	image_to_write = None
	if video_folder != None:
		if network_input == 'image':
			image_to_write = screen
		elif network_input == 'heat_map':
			image_to_write = heatmap
		elif network_input == 'both' :

			image_to_write = np.zeros((84,168,3))
			image_to_write[:,:84,:] = screen
			image_to_write[:,84:, :] = heatmap

		cv2.imwrite(video_folder + '/' + str(video_count).zfill(5) + '.png', image_to_write)

	return screen, heatmap 

def storeSaliencyImage():



def createVideo(main_folder, frame_count):
	print('Creating Video')

	fps = 4.0
	first_image = cv2.imread(main_folder + '/' + str(0).zfill(5) + '.png')
	frame_size = cvGetSize(first_image)
	writer = cvCreateVideoWriter(main_folder + "/out.avi", CV_FOURCC('F', 'L', 'V', '1'), fps, frame_size, True)

	for i in range(frame_count + 1):
		cvWriteFrame(writer, cv2.imread(main_folder + '/' + str(i).zfill(5) + '.png'))

	cvReleaseVideoWriter(writer)
	print('Video created')

def start_testing(scenario, network, path, network_input='image', heatmap_type='simple', action_space=1, args):

    print ("Initializing doom...")
    doom_env = DoomEnv(args.width, args.height)
	game = DoomGame()
	game.load_config("../../examples/config/" + scenario + ".cfg")
	game.set_mode(Mode.ASYNC_PLAYER)
	game.set_window_visible( True )
	game.set_render_weapon(True)
	game.init()
	print ("Doom initialized.")

	n = game.get_available_buttons_size()
	actions = []
	if action_space == 1:
		for perm in it.product([0, 1], repeat=n):
	    		actions.append(list(perm))

	elif action_space == 0:
		action = [0] * n
		actions.append(action)
		for i in range(0,n):
			action = [0] * n
			action[i] = 1
			actions.append(action)
	elif action_space == 2:

		actions = [[0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,1,0],[1,0,0,0,0,1,0,0,0], [0,0,0,0,0,1,1,0,0], [1,0,0,0,1,0,0,0,0], [1,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,1,0], [1,0,1,0,0,0,0,0,0], [1,0,0,0,1,0,1,0,0], [1,0,0,0,0,1,0,1,0], [0,0,0,1,0,1,0,0,0], [0,0,0,0,1,0,1,0,0], [1,0,0,0,1,0,0,0,0], [1,0,0,0,0,1,1,0,0] ]

		for i in range(1, len(actions) + 1):
			#check if actions are correct
			if actions[i-1] != spectator_action_mapping[i]:
				print('ERROR in action encoding')
				sys.exit()

	print('Total actions (' + str(len(actions)) + ')' + str(actions))
	skiprate = 10
	episodes_to_watch = 10

	steps = 0

	agent = torch.load("../../../" + path + "net-"+ str(network)+".t7")
	print(agent.network)
	print('Best agent Network is ' + str(agent.bestNetworkEpoch))

	user_response = raw_input('Do you want to use the best network ? (yes/no)')
	if user_response == 'yes':
		print('Loading best network')
		agent.loadBestNetwork(agent)
	elif user_response == 'no':
		print('Using this network')
	else:
		print('BAD INPUT')	
		sys.exit()


	agent.bestq = 0
	action_index = 2
	done = False
	episodes = 10
	sleep_time = 0.5
	running_time = 0


	total_kits = []
	total_rewards = []

	living_reward = False
	if scenario == 'deadly_corridor':
		print('Deadly Corridor scenario')
		living_reward = True
	elif scenario == 'deathmatch':
		print('Deathmatch')
		living_reward = False
	elif scenario == 'take_cover' or scenario == 'health_gathering':
		living_reward = True

	print('Checking Network : ' + str(network))

	mean_ticks = []
	mean_kills = []

	if 'doom' in path:
		print('Using doom epsilon')
		testing_ep = 0 
	elif 'dqn' in path:
		testing_ep = 0.05
	else:
		testing_ep = 0.05

	for i in range(episodes):

	    print("Episode #" + str(i+1))

	    video_folder, frame_count = None, 0
	    if args.create_video == 1:
	    	if args.create_video == 1 and not os.path.isdir(args.path + args.video_title + '-' +  str(episodes)):
				os.makedirs(args.path + args.video_title + '-' +  str(episodes))
				video_folder = args.path + args.video_title + '-' +  str(episodes)
			else:
				video_folder = args.path + args.video_title + '-' +  str(episodes)


	    game.new_episode()
	    t = 1
	    reward = game.get_total_reward()
	    screen, heatmap = getScreen(game, doom_env, network_input, heatmap_type,video_folder,frame_count, args)

	    terminal = game.is_episode_finished()

	    s1_variables = game.get_state().game_variables
	    previous_health     = s1_variables[0]
	    previous_kill_count = s1_variables[1]
	    previous_frag_count = s1_variables[2]
	    previous_ammo       = s1_variables[4]
	    previous_armor      = s1_variables[5]


	    kits = 0
	    total_episode_reward = 0
	    total_tics = 0
	    kills  = 0

	    total_shaping_reward = 0

	    while not game.is_episode_finished():

			t += 1
			frame_count += 1
			action_index = agent.perceive(agent, reward, screen, terminal, True,testing_ep)
			reward = game.make_action(actions[action_index - 1], skiprate + 1)

			
			if scenario == 'deadly_corridor':

				if reward < 0:
					reward = -80
				elif reward > 0:
					reward = 80
				else:
					reward = 0
			else:
				reward = 0

		  
			total_tics += skiprate + 1
			if game.is_episode_finished() == False:

				screen, heatmap = getScreen(game, doom_env, network_input, heatmap_type,video_folder,frame_count, args)
				s1_variables = game.get_state().game_variables

				if scenario == 'deadly_corridor' :

					reward += rewardDeadlyCorridor(previous_health, s1_variables[0], previous_frag_count, s1_variables[2], previous_kill_count, s1_variables[1],previous_ammo, s1_variables[4], 'difference')
				elif scenario == 'deathmatch':

					reward += rewardDeathMatch(previous_health, s1_variables[0], previous_frag_count, s1_variables[2], previous_kill_count, s1_variables[1],previous_ammo, s1_variables[4], s1_variables[3], previous_armor, s1_variables[5], 'difference')
					simple_reward_for_each_training_episode = s1_variables[1]

				elif scenario == 'take_cover' or scenario == 'defend_the_line':
						reward += calculateNewRewardDefendTheLine(previous_health, s1_variables[0], previous_frag_count, s1_variables[2], previous_kill_count, s1_variables[1], 'difference')
						if scenario == 'defend_the_line':
							simple_reward_for_each_training_episode = s1_variables[1]

				elif scenario == 'health_gathering':
		
		    		reward += calculateNewReward(previous_health, s1_variables[0], 'difference')
					if scenario == 'health_gathering':
						current_shaping_reward = doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
						current_shaping_reward = current_shaping_reward - total_shaping_reward
						total_shaping_reward += current_shaping_reward		

		
						if current_shaping_reward == 100:
							kits += 1

				
				previous_health =     s1_variables[0]
				previous_kill_count = s1_variables[1]
				previous_frag_count = s1_variables[2]
				previous_ammo       = s1_variables[4]
				previous_armor      = s1_variables[5]

				kills = previous_kill_count

				if living_reward == True:
					reward += skiprate + 1


				reward *= reward_scale
				total_episode_reward += reward
			else:
				final_game_reward = game.get_last_reward()
				if final_game_reward < 0:
					reward += -100
				reward *= reward_scale
				total_episode_reward += reward


			if args.store_saliency == 1:
				#get saliency and put it on image
				

		terminal = game.is_episode_finished()
	
	    sleep(sleep_time)

	    total_reward = total_episode_reward
	    mean_ticks.append(total_tics)
	    total_rewards.append(total_reward)
	    mean_kills.append(kills)

	    if video_folder != None:
	    	create_video(video_folder, video_count)

	    if scenario == 'health_gathering':
			total_kits.append(kits)

	    print("Total tics = " + str(total_tics))
	    if scenario == 'deadly_corridor' or scenario == 'deathmatch' or scenario == 'defend_the_line':
	    	print('Total game kills = ' + str(kills))
	    elif scenario == 'health_gathering':
			print('Total kits collected = ' + str(kits))
		    print('Total game reward = ' + str(total_reward))       
		    print("=====================")

	print('Final mean tics = ' + str(np.mean(mean_ticks)))
	if scenario == 'deadly_corridor' or scenario == 'deathmatch' or scenario == 'defend_the_line':
		print('Final mean kills = ' + str(np.mean(mean_kills)))
	elif scenario == 'health_gathering':
		print('Final mean kits collected ' + str(np.mean(total_kits)))
	print('Final mean score = ' + str(np.mean(total_rewards)))
	game.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default=None, help='Specify scenario for testing')
    parser.add_argument('--net', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--actions', type=int, default=1)
    parser.add_argument('--network_input', type=str, default='image')
    parser.add_argument('--heatmap_type', type=str, default='simple')
    parser.add_argument('--create_video', type=int, default=0)
    parser.add_argument('--video_title', type=str,default=None)
    parser.add_argument('--width', type=int, default=84)
    parser.add_argument('--height', type=int, default=84)
    parser.add_argument('--saving_dir', type=str, default=None)
    parser.add_argument('--display', type=int, default=1)
    parser.add_argument('--store_saliency', type=int, default=0)
    parser.add_argument('--heatmap_network_type', type=int, default=1)
    

    args = parser.parse_args()
    if args.scenario is None or args.net is None or args.path is None:
		print('Error : Please enter the arguments properly !')
    else:



		print('Started testing for ' + args.scenario + ' using the network ' + args.net + ' in ' + args.path)
		start_testing(args.scenario, args.net, args.path, args.network_input, args.heatmap_type, args.actions, args)

if __name__ == '__main__':
    main()

