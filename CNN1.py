#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pygame as pyg
import random


import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices

from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

import matplotlib.pyplot as plt # Display graphs
import matplotlib.image as mimg
from collections import deque# Ordered collection with ends


# In[2]:


possible_actions = np.array(np.identity(4,dtype=int).tolist())


# In[3]:


def preprocess_frame(frame):
    # Greyscale frame 
    gray = rgb2gray(frame)
    
    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    #cropped_frame = gray[8:-12,4:-12]
    
    # Normalize Pixel Values
    normalized_frame = gray/255.0
    
    # Resize
    # Thanks to Mikołaj Walkowiak
    preprocessed_frame = transform.resize(normalized_frame, [110,84])
    
    return preprocessed_frame # 110x84x1 frame


# In[4]:


stack_size = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames


# In[5]:


### MODEL HYPERPARAMETERS
state_size = [110, 84, 4]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels) 
action_size = 4 #8 ossible actions
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 50            # Total episodes for training
max_steps = 50000              # Max possible steps in an episode
batch_size = 64                # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.00001           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9                    # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4                 # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True


# In[6]:


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            ELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")            
            
            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")
            
            self.output = tf.layers.dense(inputs = self.fc, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)
            

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


# In[7]:


# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)


# In[8]:


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]


# In[9]:




"""
This function will do the part
With ϵϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        choice = random.randint(1,len(possible_actions))-1
        action = possible_actions[choice]
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[choice]
                
                
    return action, explore_probability


# In[10]:


pyg.init()

clock = pyg.time.Clock()

# FONT -
fontScore = pyg.font.SysFont('comicsans', 30, True, True)
fontEnd = pyg.font.SysFont('comicsans', 70, True)

winSize = (960, 810)
sptSize = (128, 128)


# In[11]:


# defining class of our ship
class ship:
    static = [pyg.transform.scale(pyg.image.load('/home/ritik/Desktop/L1.png'), sptSize),
              pyg.transform.scale(pyg.image.load('/home/ritik/Desktop/L1.png'), sptSize)]
    flyRight = [pyg.transform.scale(pyg.image.load('/home/ritik/Desktop/L1.png'), sptSize),
                pyg.transform.scale(pyg.image.load('/home/ritik/Desktop/L1.png'), sptSize)]
    flyLeft = [pyg.transform.scale(pyg.image.load('/home/ritik/Desktop/L1.png'), sptSize),
               pyg.transform.scale(pyg.image.load('/home/ritik/Desktop/L1.png'), sptSize)]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dir = 0
        self.vel = 15
        self.width = 90     # not image width but the width of the rectangle around it
        self.moveCount = 0  # to switch between sprites
        self.health = 200

    def draw(self, win):
        if self.moveCount == 30:
            self.moveCount = 0
        else:
            self.moveCount += 1

        if self.dir == 0:
            win.blit(self.static[self.moveCount % 2], (self.x, self.y))
        elif self.dir == 1:
            win.blit(self.flyRight[self.moveCount % 2], (self.x, self.y))
        elif self.dir == -1:
            win.blit(self.flyLeft[self.moveCount % 2], (self.x, self.y))
        pyg.draw.rect(win, (0, 255, 0), (self.x + 15, self.y + self.width + 20, self.health/10, 10), 0)
        pyg.draw.rect(win, (10, 255, 0), (self.x + 15, self.y + self.width + 20, self.width, 10), 1)


# In[12]:


class Asteroids:

    def __init__(self, x, s):
        self.x = x
        self.y = -s
        self.vel = 6 - (s-100)//50
        self.width = s - 20
        self.pop = False
        self.size = (s, s)
        self.health = s*2
        self.img = pyg.transform.scale(pyg.image.load('/home/ritik/Desktop/e.png'), self.size)

    def motion(self):
        if self.y < 810:
            self.y += self.vel
        else:
            self.pop = True

    def draw(self, win):
        self.motion()
        win.blit(self.img, (self.x, self.y))
        # pyg.draw.rect(win, (255, 0, 0), (self.x+10, self.y, self.width, self.width), 2)


# In[13]:


class shots():
    size = (48, 48)
    image = pyg.transform.scale(pyg.image.load('/home/ritik/Desktop/bullet.png'), size)

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vel = 16
        self.pop = False
        self.width = 20

    def motion(self):
        if self.y + 10 > 0:
            self.y -= self.vel
        else:
            self.pop = True

    def draw(self, win):
        self.motion()
        win.blit(self.image, (self.x, self.y))
        # pyg.draw.rect(win, (255, 0, 0), (self.x + 14, self.y, self.width, self.width), 2)


# In[14]:


# SOUND EFFECTS -
#hitSound = pyg.mixer.Sound('resources/destroy.wav')
#fireSound = pyg.mixer.Sound('resources/shoot.wav')
#music = pyg.mixer.music.load('resources/music.mp3')
#pyg.mixer.music.play(-1)
#pyg.mixer.music.set_volume(0.5)

# Window setup

win = pyg.display.set_mode(winSize)
bg = pyg.image.load('/home/ritik/Desktop/b.jpg')
pyg.display.set_caption('TheGame')
startLocn = ((winSize[0] - sptSize[0])//2, winSize[1] - sptSize[1] - 10)
player = ship(startLocn[0], startLocn[1])
bullets = []
asteroids = []
score = 0

done = False
play = True
lastPosition = 100
bulletInterval = 5
deathcount = 0
asteroidInterval = 18


# In[15]:


def Reset():
    global player,bullets,startLocn,asteroids,score,done,play,lastPosition,bulletInterval,asteroidInterval
    score = 0
    restartCounter = 100
    delayCounter = 25
    done = False
    #start = True
    lastPosition = 100
    bulletInterval = 5
    asteroidInterval = 18
    player = ship(startLocn[0], startLocn[1])
    bullets = []
    asteroids = []
    screen, missed = redrawWin()
    return screen


# In[16]:


def redrawWin():
    global player,bullets,asteroids,score,done,play,lastPosition,bulletInterval,asteroidInterval
    win.blit(pyg.transform.scale(bg, winSize), (0, 0))
    #if not start:
        #textStart = fontEnd.render('PRESS SPACE TO START', 2, (255, 0, 0))
        #win.blit(textStart, (winSize[0] / 2 - 320, winSize[1] / 2 - 50))
    #elif end:
        #text1 = fontEnd.render('GAME OVER', 2, (255, 0, 0))
        #text2 = fontEnd.render('RESTARTING SOON', 2, (255, 0, 0))
        #win.blit(text1, (winSize[0] / 2 - 200, winSize[1] / 2 - 50))
        #win.blit(text2, (winSize[0] / 2 - 250, winSize[1] / 2))
    #else:
    text = fontScore.render('Score - ' + str(score), 2, (27, 229, 54))
    win.blit(text, (winSize[0] - 150, winSize[1] - 50))
    if not done:
       # if start:
       #     player.dir = 0
        player.draw(win)
    for bullet in bullets:
        if bullet.pop:
            bullets.pop(bullets.index(bullet))
        else:
            bullet.draw(win)
    missed = False
    for asteroid in asteroids:
        if asteroid.pop:
            asteroids.pop(asteroids.index(asteroid))
            missed = True
        else:
            asteroid.draw(win)

    pyg.display.update()
    return pyg.surfarray.array3d(win).swapaxes(0,1),missed


# In[17]:


def Step(action):
    global player,bullets,asteroids,score,end,done,deathcount,start,play,lastPosition,bulletInterval,asteroidInterval
    action = tuple(action)
    if action == (1,0,0,0) and player.x > 0:
        player.dir = -1
        player.x -= player.vel
    elif action == (0,1,0,0) and player.x + player.width < winSize[0]:
        player.dir = 1
        player.x += player.vel
    elif action == (0,0,1,0):
        player.dir=0
        if bulletInterval > 4:
             #   fireSound.play()
            bulletInterval = 0
            bullets.append(shots(player.x + (player.width//2)-6, player.y))
        else:
            bulletInterval += 1
    reward=0
    for asteroid in asteroids:
        if max(abs(asteroid.x + 10 + asteroid.width // 2 - player.x - 15 - player.width // 2),
               abs(asteroid.y + asteroid.width // 2 -
                   player.y - 10 - player.width // 2)) < ((asteroid.width + player.width) // 2 - 10):
            # print('You hit an asteroid')
           # if not start and not end:
            if player.health - 100 > 0:
                player.health -= 100
                asteroid.pop = True
            else:
                player.health = 0
            reward-=1
        elif (asteroid.y + asteroid.width) > player.y and (asteroid.x + asteroid.width // 2) < ((player.x +player.width //2) +3 * player.width) and (asteroid.x + asteroid.width // 2) > ((player.x +player.width //2) -3 * player.width) :
        	reward+=0.5        
        for bullet in bullets:
            if bullet.y <= asteroid.y + asteroid.width - 5 and bullet.y > asteroid.y:
                if bullet.x > asteroid.x - 5 and bullet.x < asteroid.x + asteroid.width - 5:
                    bullet.pop = True
                    asteroid.health -= 150
                    score += 10
                    reward+=0.5
        if asteroid.health < 0:
            #hitSound.play()
            asteroid.pop = True
            reward+=1
    
    if asteroidInterval > 19:
        asteroidInterval = 1
        size = random.randint(100, 200)
        pos_x = random.randint(0, 800)
        if abs(lastPosition - pos_x) > 300 and (pos_x - size // 2) < winSize[0]:
            lastPosition = pos_x
            asteroids.append(Asteroids(pos_x, size))
    else:
        asteroidInterval += 1    
    if player.health == 0:
            done = True
            deathcount+=1
    screen,missed = redrawWin()
    if missed == True:
        reward-=1
    return screen,reward,done


# In[18]:


# Instantiate memory
memory = Memory(max_size = memory_size)
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        state = Reset()
        
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    # Get the next_state, the rewards, done by taking a random action
    choice = random.randint(1,len(possible_actions))-1
    action = possible_actions[choice]
    next_state, reward, done = Step(action)
    
    #env.render()
    
    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    
    
    # If the episode is finished (we're dead 3x)
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        #deathcount = 0
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        # Start a new episode
        state = Reset()
        
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        # Our new state is now the next_state
        state = next_state


# In[ ]:





# In[19]:




# Setup TensorBoard Writer
writer = tf.summary.FileWriter("~/tensorboard/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()


# In[ ]:


# Saver will help us to save our model
saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0
        
        rewards_list = []
        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            state = Reset()
            
            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            
            while step < max_steps:
                step += 1
                
                #Increase decay_step
                decay_step +=1
                
                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
                
                #Perform the action and get the next_state, reward, and done information
                next_state, reward, done = Step(action)
                
              #  if episode_render:
              #      env.render()
                
                # Add the reward to total reward
                episode_rewards.append(reward)
                
                # If the game is finished
                if done:
                    
                    # The episode ends so no next state
                    next_state = np.zeros((110,84), dtype=np.int)
                    
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(total_reward),
                                  'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))

                    rewards_list.append((episode, total_reward))

                    # Store transition <st,at,rt+1,st+1> in memory D
                    memory.add((state, action, reward, next_state, done))

                else:
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                
                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state
                    

                ### LEARNING PART            
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state 
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                        feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb})

                # Write TF Summaries
        
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                       DQNetwork.target_Q: targets_mb,
                                                       DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")


# In[21]:


with tf.Session() as sess:
    total_test_rewards = []
    
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    
    for episode in range(1):
        total_rewards = 0
        
        state = Reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
        print("****************************************************")
        print("EPISODE ", episode)
        
        while True:
            # Reshape the state
            state = state.reshape((1, *state_size))
            # Get action from Q-network 
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice]
            
            #Perform the action and get the next_state, reward, and done information
            next_state, reward, done = Step(action)
#            env.render()
            
            total_rewards += reward

            if done:
                print ("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break
                
                
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
            
#    env.close()


# In[ ]:





# In[ ]:





# In[ ]:




