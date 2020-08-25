from vizdoom import *
import numpy as np
import itertools

class simulator():
    
	def __init__(self,a_size):
		game = DoomGame()
		game.set_doom_scenario_path(".\\vizdoom\\scenarios\\health_gathering.wad") #This corresponds to the simple task we will pose our agent
		game.set_doom_map("map01")
		game.set_screen_resolution(ScreenResolution.RES_160X120)
		game.set_screen_format(ScreenFormat.GRAY8)
		game.set_render_hud(False)
		game.set_render_crosshair(False)
		game.set_render_weapon(True)
		game.set_render_decals(False)
		game.set_render_particles(False)
		game.add_available_button(Button.TURN_LEFT)
		game.add_available_button(Button.TURN_RIGHT)
		game.add_available_button(Button.MOVE_FORWARD)
		#game.add_available_button(Button.MOVE_BACKWARD)
		#game.add_available_button(Button.MOVE_LEFT)
		#game.add_available_button(Button.MOVE_RIGHT)
		#game.add_available_button(Button.ATTACK)
		
		game.add_available_game_variable(GameVariable.HEALTH)
		#game.add_available_game_variable(GameVariable.POSITION_X)
		#game.add_available_game_variable(GameVariable.POSITION_Y)
		game.set_episode_timeout(5000)
		#game.set_episode_start_time(10)
		game.set_window_visible(False)
		game.set_sound_enabled(False)
		game.set_living_reward(-0.01)
		game.set_mode(Mode.PLAYER)
		game.init()
		self.actions = np.identity(a_size,dtype=bool).tolist()
		self.game = game
		
	def initialize(self):
		self.game.new_episode()

	def fetch(self):
		f = self.game.get_state().screen_buffer
		return f

	def move(self,a):
		temp = self.game.make_action(self.actions[a])
		r = self.game.get_game_variable(HEALTH)	
		d = self.game.is_episode_finished()
		return r,d
    
