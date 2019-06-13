# Using pybullet as enviornment.

import numpy as np
import os

# Setting the hyperparameters
class HyperParameters():

    def __init__(self):
        self.no_of_steps_per_epoch = 1000
        # Max number of actions per episode is episode length
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.no_of_directions = 16
        self.no_of_best_directions = 16
        assert self.no_of_best_directions <= self.no_of_directions
        # standard_deviation of normal distribution is also called as noise. 
        self.standard_deviation_for_normal_distribution = 0.03
        self.seed = 1  # For enviornment configurations
        self.env_name = 'HalfCheetahBulletEnv-v0'

# Normalization of states
class Normalizer():

    def __init__(self,no_of_inputs):
        self.n = np.zeros(no_of_inputs)
        self.mean = np.zeros(no_of_inputs)
        self.mean_difference = np.zeros(no_of_inputs) # i.e (x - u)
        self.variance = np.zeros(no_of_inputs)
    
    # for calculation of real time mean and variance
    def observe(self,inputs):
        self.n += 1
        last_mean = self.mean.copy() # For updation of variance
        self.mean += (inputs - last_mean) / self.n
        self.variance = (self.mean_difference / self.n).clip(min = 1e-2)
    
    def normalize(self,inputs):
        observed_mean = self.mean
        observed_standard_deviation = np.sqrt(self.variance)
        return (inputs - observed_mean) / observed_standard_deviation

# Building the AI
class Policy():

    def __init__(self,input_size,output_size):
        # theta will be the matrix containing the actual_weights 
        self.theta = np.zeros((output_size,input_size)) # Because it will be on left_side of multiplication , for right_side multiplication shape will be ((input_size,output_size))

    # performing (weights +- noise*delta)*input 
    def evaluate(self,input_data,delta = None,direction = None):
        if direction is None:
            return self.theta.dot(input_data)
        elif direction == "positive":
            return (self.theta + hp.standard_deviation_for_normal_distribution * delta).dot(input_data)
        else:
            return (self.theta - hp.standard_deviation_for_normal_distribution * delta).dot(input_data)
    
    def creating_sample_deltas(self):
        # In randn,n is for normal distribution 
        # this function accepts dimensions in an unpacked manner i.e np.random.randn(self.theta.shape[0],self.theta.shape[1])  
        return [ np.random.randn(*self.theta.shape) for _ in range(hp.no_of_directions) ]

    # Actual updation of weights
    def update(self,rollouts,sigma_r):
        # Here rollouts contains the triplets and each triplet is made up of:
        # 1) reward_for_postive_direction 2)reward_for_negative_direction 3)actual_delta
        # sigma_r is the standard deviation of the reward
        step = np.zeros(self.theta.shape) # here np.zeros require tuple
        for r_pos , r_neg , delta in rollouts:
            step += (r_pos - r_neg) * delta
        
        self.theta += hp.learning_rate / (hp.no_of_best_directions * sigma_r) * step

def explore(env,normalizer,policy,direction = None , delta = None):
    state = env.reset()
    done = False
    num_of_actions_played = 0
    total_reward = 0

    while not done and num_of_actions_played < hp.episode_length:
        normalizer.observe(state)
        normalized_state = normalizer.normalize(state)
        action = policy.evaluate(state,delta,direction)
        state , reward , done = env.step(action)

        # Since the reward can be sometimes be very large or very small this could affect the cumalative_reward
        # So it should be normalized.
        # if it is too positive it is set to 1,elif it is too negative it is set to -1.
        reward = max( min(reward , 1), -1 )
        total_reward += reward
        num_of_actions_played += 1
    
    return total_reward

# Training the AI
def train(env,policy,normalizer,hp):

    for step in range(hp.no_of_steps_per_epoch):

        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.creating_sample_deltas()
        postive_rewards = [0] * [hp.no_of_directions] # Creating list of zeros of length equal to no_of_directions
        negative_rewards = [0] * [hp.no_of_directions] 

        # Creating the positive rewards in the positive directions
        for k in range(hp.no_of_directions):
            postive_rewards[k] = explore(env,normalizer,policy,direction="postive",delta=deltas[k])
        
        
        # Creating the negative rewards in the positive directions
        for k in range(hp.no_of_directions):
            negative_rewards[k] = explore(env,normalizer,policy,direction="negative",delta=deltas[k])
        
        # Gathering all the postive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards) # Concatinating both the lists
        std_of_rewards = all_rewards.std() 

        # Sorting the  rollouts by the max(r_pos,r_neg) and selecting the best directions
        scores = {k:max(r_pos,r_neg) for k in (r_pos,r_neg) in enumerate(zip(postive_rewards,negative_rewards))}
        # Getting the keys upto no_of_best_directions which have max rewards
        order = sorted(scores.keys(), key = lambda x:scores[x])[:hp.no_of_best_directions]
        rollouts = [(postive_rewards[k],negative_rewards[k],deltas[k]) for k in order]

        # Updating the policy
        policy.update(rollouts,std_of_rewards)

        # Printing the final reward
        reward_evaluated = explore(env,normalizer,policy)
        print('Step : ',step,'Reward : ',reward_evaluated)

# Creating the directory in which all the results will be stored (as videos)
def mkdir(base,name):
    path = os.path.join(base,name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    return path

working_dir = mkdir('exp','results')
monitor_dir = mkdir(working_dir , 'monitor')

hp = HyperParameters()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)

# Saving the video by using wrappers package of gym
# force = True means if any warning occurs show it on the console
env = wrappers.Monitor(env,monitor_dir,force = True) 
no_of_inputs = env.observation_space.shape[0]
no_of_outputs = env.action_space.shape[0]
policy = Policy(no_of_inputs,no_of_outputs)
normalizer = Normalizer(no_of_inputs)
train(env,policy,normalizer,hp)
