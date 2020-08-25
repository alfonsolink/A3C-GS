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

num_workers = 6
environment = 'Doom_Basic'
set_learning_rate = 1e-4
no_of_actions = 3
clip_using_norm = True
clip_norm_magnitude = 40.0
clip_value_magnitude = 0.001
lives = 5
logfilename = 'scores_'+str(environment)+'_w'+str(num_workers)+'_standard.csv'

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

def create_network(inputs):
    net = tf.reshape(inputs,[-1,84,84,1])
    net = slim.conv2d( \
        inputs=net,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID',biases_initializer=None,activation_fn=tf.nn.relu)
    net = slim.conv2d( \
        inputs=net,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID',biases_initializer=None,activation_fn=tf.nn.relu)
    net = slim.conv2d( \
        inputs=net,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID',biases_initializer=None,activation_fn=tf.nn.relu)
    fc_h = slim.fully_connected(slim.flatten(net),256,activation_fn=tf.nn.relu)
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
            self.inputs = tf.placeholder(tf.float32,[None,s_size])
            
            self.policy, self.value = create_network(self.inputs)
            
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -0.5 * tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = self.value_loss + self.policy_loss - self.entropy * 0.01
                
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                
                self.gradients = tf.gradients(self.loss,local_vars)
                self.gradients = modify_gradients(self.gradients)
                #self.gradients,_ = tf.clip_by_global_norm(self.gradients,40.0)
                self.apply_grads = trainer.apply_gradients(zip(self.gradients,global_vars))
                
                
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def discountx(r,gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        #if r[t] != 0.0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

global_max = np.zeros(num_workers)

class Worker():
    def __init__(self,name,trainer,cfg):
        self.name = "worker_" + str(name)
        self.number = name
        self.trainer = trainer
        self.cfg = cfg
        self.local_AC = AC_Network(self.cfg.s_size,self.cfg.a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)
        self.env = simulator(a_size)
        self.experience = experience(self.cfg)

    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        #t0 = time.time()
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discountx(self.rewards_plus,0.99)
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + 0.99*discounted_rewards[1:] - self.value_plus[:-1]
        feed_dict = {self.local_AC.target_v:discounted_rewards[:-1],
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
        v_l,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        sess.run(self.update_local_ops)
        #print (time.time() - t0)
        return v_l / len(rollout)
    
    def work(self,sess,coord,saver):
        total_steps = 0
        print ("Starting worker " + str(self.number))
        stats = 0.0
        losers = 0.0
        steps = 0.0
        global lives
        
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_step_count = 0
                d = False
                self.experience.reset()
                self.env.initialize()
                s = self.env.fetch()
                s = process_frame(s)
                
                while d == False:
                    a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value], 
                        feed_dict={self.local_AC.inputs:[s]})
                    a_dist = a_dist.reshape(-1)
                    a = np.random.choice(a_dist,p=a_dist)
                    a = np.argmax(a_dist == a)
                    r,d = self.env.move(a)
                    r = r/100.0

                    if r == 1.0: stats += 1.0
                    if d == False:
                        s1 = self.env.fetch()
                        show = s1
                        s1 = process_frame(s1)
                    else:
                        steps += 1.0
                        s1 = s
                        
                    self.experience.add_experience([s,a,r,s1,d,v[0,0]]) #add to experience buffer
                    
                    if self.number == 0:
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
    trainer = tf.train.AdamOptimizer(learning_rate=set_learning_rate)
    #trainer = tf.train.MomentumOptimizer(learning_rate=1e-3,momentum=0.9,use_nesterov=False)
    #trainer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
    master_network = AC_Network(cfg.s_size,cfg.a_size,'global',None) 
    workers = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    for i in range(num_workers):
        workers.append(Worker(i,trainer,cfg))
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
            
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
