import tensorflow as tf
from time import sleep
import threading
import numpy as np
from tensorflow.contrib import slim
import time
import scipy.signal
import cv2
#import gym
import csv
from scipy.ndimage.filters import gaussian_filter1d
from simulator import simulator

num_workers = 2; global_workers = 3 #there are 2 workers under 3 sets of global parameters
environment = 'Doom_Health'
set_learning_rate = 1e-5
no_of_actions = 3
entropy_factor = 0.005
clip_using_norm = True
clip_norm_magnitude = 40.0
clip_value_magnitude = 0.001
lives = 5
logfilename = 'scores_'+str(environment)+'_w'+str(num_workers)+'_g'+str(global_workers)+'_GS.csv'

def process_frame(x): #image preprocessing following http://karpathy.github.io/2016/05/31/rl/
    #s = frame[10:-10,30:-30]
    s = cv2.resize(x,(84,84))
    #s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s.reshape(-1)

class experience(): #experience buffer for storing trajectories and rewards
    def __init__(self,cfg):
        self.episode_buffer = []
    def add_experience(self,x):
        self.episode_buffer.append(x)
    def reset(self):
        self.episode_buffer = []

class cfg(): #cfg class for hyperparameters
    def __init__(self):
        self.gamma = 0.99 #discount rate
        self.s_size = 7056*2 #input to network(flattened image vector)
        self.a_size = no_of_actions #number of actions (Pong-specific)

cfg = cfg() #initialize cfg class
a_size = cfg.a_size

#normalized_columns_initializer lifted from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        out[out > 1.0] = 1.0; out[out < -1.0] = -1.0
        return tf.constant(out)
    return _initializer

def modify_gradients(x):
    x = [tf.where(tf.is_nan(grad),tf.zeros_like(grad),grad) for grad in x]
    x = [tf.where(tf.equal(grad,np.inf),tf.zeros_like(grad),grad) for grad in x]
    x = [tf.where(tf.equal(grad,-np.inf),tf.zeros_like(grad),grad) for grad in x]
    if clip_using_norm == True:
        x,_ = tf.clip_by_global_norm(x,clip_norm_magnitude) #clipping operation using global norm
    else:
        x = [tf.clip_by_value(grad,-clip_value_magnitude,+clip_value_magnitude) for grad in x] #alternative clipping operation using clip by value
    return x

def create_network(inputs): #network trunk using Rainbow architecture
    net = tf.reshape(inputs,[-1,84,84,1])
    net = slim.conv2d( \
        inputs=net,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID',biases_initializer=None,activation_fn=tf.nn.relu)
    net = slim.conv2d( \
        inputs=net,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID',biases_initializer=None,activation_fn=tf.nn.relu)
    net = slim.conv2d( \
        inputs=net,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID',biases_initializer=None,activation_fn=tf.nn.relu)
    fc_h = slim.fully_connected(slim.flatten(net),256,activation_fn=tf.nn.relu)
    #set up dueling network for policy and value networks
    policy = slim.fully_connected(fc_h,a_size,
        activation_fn=tf.nn.softmax,
        weights_initializer=normalized_columns_initializer(0.01),
        biases_initializer=None)
    value = slim.fully_connected(fc_h,1,
        activation_fn=None,
        weights_initializer=normalized_columns_initializer(0.01),
        biases_initializer=None)
    return policy, value
    

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        s_size = s_size/2
        
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(tf.float32,[None,s_size]) #placeholder for flattened image inputs
            self.policy, self.value = create_network(self.inputs) #create policy / value networks
        
        if 'global' in scope:
            self.block = tf.Variable(initial_value=[True],dtype=tf.bool,trainable=False,name='blocker') #create lock (as Variable)
            
        if 'global' not in scope:
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32) #feed trajectory actions
            self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None],dtype=tf.float32) #feed trajectory MSE targets
            self.advantages = tf.placeholder(shape=[None],dtype=tf.float32) #feed advantage targets
            
            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1]))) #value loss
            self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy)) #compute entropy
            self.policy_loss = -0.5 * tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages) #policy loss
            self.loss = self.value_loss + self.policy_loss - self.entropy * entropy_factor #mpound loss with 0.005 entropy weight
            
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope) #local worker parameter set
            own_global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_'+str(scope[-1])) #assigned global parameter set
            
            self.others_list = [int(x) for x in range(global_workers) if x != int(str(scope[-1]))] 
            other_global_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_'+str(i)) for i in self.others_list] #other global parameter sets
            
            self.gradients = modify_gradients(tf.gradients(self.loss,local_vars)) #get local gradients
            self.apply_own_grads = trainer.apply_gradients(zip(self.gradients,own_global_vars)) #handle to apply gradients to assigned global parameter set
            
            self.feed_gradients = [tf.reshape(tf.placeholder(shape=[None],dtype=tf.float32),grad.shape) for grad in self.gradients] #placeholder to get gradients from self.apply_own_grads
            self.apply_other_grads = [trainer.apply_gradients(zip(self.feed_gradients,other_global_vars[i])) for i in range(len(self.others_list))] #handle to apply gradients to other global parameter sets
            
            self.update_local_ops = update_target_graph('global_'+str(scope[-1]),scope) #handle to update worker parameters
            self.transfer_global = [update_target_graph('global_0','global_'+str(i)) for i in range(global_workers)] #handle to set all parameters as equal (used only once)
            
            self.block_vars = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'blocker' in x.name] #handle to activate lock
            self.block_global = [tf.assign(self.block_vars[i],[False]) for i in range(global_workers)] #handle to activate lock
            self.unblock_global = [tf.assign(self.block_vars[i],[True]) for i in range(global_workers)] #handle to deactivate lock       
              

def update_target_graph(from_scope,to_scope): #create handles to transfer weights
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discountx(r,gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        #if r[t] != 0.0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!) as per http://karpathy.github.io/2016/05/31/rl/
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

global_max = np.zeros(num_workers)

class Worker():
	def __init__(self,global_assignment,name,trainer,cfg):
		self.name = str(name)+"_worker_" + str(global_assignment)
		self.global_assignment = self.name[-1] #get global assignment index
		self.trainer = trainer #get initialized Adam optimizer
		self.cfg = cfg
		self.local_AC = AC_Network(self.cfg.s_size,self.cfg.a_size,self.name,trainer) #set up worker
		self.env = simulator(a_size)
		self.experience = experience(self.cfg)
		
	def train(self,rollout,sess,gamma,bootstrap_value): #some parts of train function are lifted from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
		rollout = np.array(rollout)
		observations = rollout[:,0] #trajectory states
		actions = rollout[:,1] #trajectory actions
		rewards = rollout[:,2] #trajectory rewards
		next_observations = rollout[:,3]
		values = rollout[:,5] #trajectory value functions
		self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value]) #bootstrap value is computed value function at the 32th index of the trajectory
		discounted_rewards = discountx(self.rewards_plus,0.99) #discounted rewards using actual rewards + boostrap vaue
		self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + 0.99*discounted_rewards[1:] - self.value_plus[:-1] #compute advantage targets
		
		
		feed_dict = {self.local_AC.target_v:discounted_rewards[:-1],
			self.local_AC.inputs:np.vstack(observations),
			self.local_AC.actions:actions,
			self.local_AC.advantages:advantages}
		
		grads, _ = sess.run([self.local_AC.gradients,self.local_AC.apply_own_grads],feed_dict=feed_dict) #get gradients and perform SGD updates on assigned global parameters
		sess.run(self.local_AC.update_local_ops) #update worker parameters
		
		self.other_ids = list(range(global_workers))
		del self.other_ids[int(self.global_assignment)]
		self.block_stats = sess.run(self.local_AC.block_vars) #get lock status
		self.valid_updates = list(np.array(self.block_stats))
			
		while not all([v == True for v in self.valid_updates]): #loop until all locks are de-activated
			self.block_stats = sess.run(self.local_AC.block_vars)
			self.valid_updates = list(np.array(self.block_stats))
		
		feed_dict = {k:v for (k,v) in zip(self.local_AC.feed_gradients,grads)} 
		for i in range(global_workers): sess.run(self.local_AC.block_global[int(i)]) #activate all locks
		for i in range(len(self.other_ids)): sess.run(self.local_AC.apply_other_grads[int(i)], feed_dict=feed_dict) #apply grads to all other global parameters
		for i in range(global_workers): sess.run(self.local_AC.unblock_global[int(i)]) #de-activate all locks
		
		
	def work(self,sess,coord,saver):
		total_steps = 0
		print ("Starting worker " + str(self.name))
		stats = 0.0
		losers = 0.0
		steps = 0.0
		global lives
		
		with sess.as_default(), sess.graph.as_default():
			
			sess.run(self.local_AC.transfer_global)
			
			while not coord.should_stop():
				
				sess.run(self.local_AC.update_local_ops)
				episode_step_count = 0
				d = False
				self.experience.reset()
				self.env.initialize()
				s = self.env.fetch()
				s = process_frame(s)
				old_h = 100.0
				
				while d == False:
					a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value], 
						feed_dict={self.local_AC.inputs:[s]})
					a_dist = a_dist.reshape(-1)
					a = np.random.choice(a_dist,p=a_dist)
					a = np.argmax(a_dist == a)
					new_h,d = self.env.move(a)
					#r = r/100.0

					if new_h > old_h: r = 1.0 #/100.0
					else: r = -1.0 #/100.0
					stats += new_h
					old_h = new_h
					if d == False:
						s1 = self.env.fetch()
						show = s1
						s1 = process_frame(s1)
					else:
						steps += 1.0
						s1 = s
						
					self.experience.add_experience([s,a,r,s1,d,v[0,0]]) #add to experience buffer
					
					if int(self.name[0]) == 0 and int(self.name[-1]) == 0:
						#cv2.imshow("",show)
						#cv2.waitKey(1)
						if total_steps % 10000 == 0: print(total_steps)
						
						if total_steps % int(50000/8) == 0:
							print("")
							print("wins     : " + str(stats*8.0))
							print("loss     : " + str(losers*8.0))
							print("steps    : " + str(steps*8.0))
							print("kill stat: " + str(stats/((steps)+1.0)))
							print("")
							with open(logfilename,'a') as f:
								writer = csv.writer(f)
								writer.writerow([stats,losers,steps,total_steps])
							
							steps = 0
							stats = 0.0
							losers = 0.0
					s = s1
					episode_step_count += 1
					total_steps += 1
					
					if len(self.experience.episode_buffer) == 32 and d != True:      
						bootstrap_value = sess.run([self.local_AC.value], feed_dict={self.local_AC.inputs:[s1]}) #get bootstrap
						self.train(self.experience.episode_buffer,sess,self.cfg.gamma,bootstrap_value[0]) #call train procedure
						self.experience.reset()
					
					if d == True:
						break
					
				if len(self.experience.episode_buffer) != 0:
					self.train(self.experience.episode_buffer,sess,self.cfg.gamma,0.0)  #call train procedure
					self.experience.reset()



if __name__ == "__main__":

    tf.reset_default_graph()
    trainer = tf.train.AdamOptimizer(learning_rate=set_learning_rate) #initialize Adam optimizer
    workers = []
    global_nets = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    for i in range(global_workers):
        agent_name = 'global_'+str(i)
        global_nets.append(AC_Network(cfg.s_size,cfg.a_size,agent_name,None)) #create global parameter sets
    
    for j in range(global_workers):
        for i in range(num_workers):
            workers.append(Worker(j,i,trainer,cfg)) #create workers for each global parameter set
    
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
            
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(sess,coord,saver)
            t = threading.Thread(target=(worker_work)) #threading operator to run multiple workers
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
