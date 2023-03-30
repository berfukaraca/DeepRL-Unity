import math
import datetime
import os, sys
import random
from tkinter.tix import MAX
from turtle import distance
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import namedtuple
from itertools import count
from IPython.display import Audio
import csv
import time
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import struct
import torchvision.transforms as T
from torchsummary import summary

import cv2




# local files
sys.path.insert(0, '../')
import pyClient
import utils
import model
from model import DoubleDQNAgent, Transition



def validation_loop(agent,environment,img_processing, cfg, val_seeds=[251,252,253,254,255]):
    # How to handle the different end signals
    RESET_UPON_END_SIGNAL = {0:False,  # Nothing happened
                             1:False,   # Box collision
                             2:False,   # Wall collision
                             3:True,    # Reached step target
                             4:True}    # Reached target

    # Set nn.module to evaluation mode
    agent.policy_net.eval()

    validation_list=[]
    seed_num=0

    total_distance_travelled=0
    total_approach_to_targets = 0
    reward_list=[]
    target_tot=0
    target_reached=0

    for seed in val_seeds:
        seed_num=seed_num+1
        
        
        # total_loss = 0 # COMMENT OUT TO REGISTER CUMULATIVE LOSS
        validation_loss=0
        wall_collisions = 0
        box_collisions = 0
        total_reward=0
        endless_loops = 0
        step_count = 0
        
        max_steps_reached=0
        action_forward=0
        action_rotateRight=0
        action_rotateLeft=0
        distance_travelled=0
        final_distancetotarget=0
        initial_dist_to_targets = 0



        # Reset environment at start of episode
        _, _, _,_= environment.setRandomSeed(seed)
        _, _, initial_dist_to_target,frame_raw= environment.reset(cfg['validation_condition'])
        # Create an empty frame stack and fill it with frames
        frame_stack = utils.FrameStack(stack_size=cfg['stack_size'] )
        frame = img_processing(frame_raw).to(agent.device)
        shape=(1,1,128,128)

        # ADD ANOTHER INPUT TO THE NETWORK
        # COMMENT OUT NEXT 2 LINES TO DELETE THE ADDITIONAL INPUT
        frame_dist=torch.ones(shape).to(agent.device)
        frame_dist[:]=initial_dist_to_target/255

        # frame_dist=torch.Tensor(frame_dist / 255.).view(1,1,IMSIZE,IMSIZE)
        

        for _ in range(cfg['stack_size'] ):
            # _, _,_, frame_raw = environment.step(0)
            state = frame_stack.update_with(frame)
        state= torch.cat((state,frame_dist),1) # add the additional input  COMMENT OUT TO DELETE THE ADDITIONAL INPUT

        # Episode starts here:
        for t in count():

            # 1. Agent performs a step (based on the current state) and obtains next state
            action = agent.select_action(state,validation=True)
            # plt.imshow(state[0,4,:,:])
            # plt.show()
            end, reward,distancetoTarget, next_state_raw = environment.step(action.item())
            agent_died = RESET_UPON_END_SIGNAL[end]            
            frame = img_processing(next_state_raw).to(agent.device)

            frame_dist[:]=distancetoTarget/255 # normalize with pixel number COMMENT OUT TO DELETE THE ADDITIONAL INPUT
            # frame_dist=torch.Tensor(frame_dist / 255.).view(1,1,IMSIZE,IMSIZE)
            next_state = frame_stack.update_with(frame) if not agent_died else None
            next_state= torch.cat((next_state,frame_dist),1) if not agent_died else None # COMMENT OUT TO DELETE THE ADDITIONAL INPUT
            if reward >= 100:
                reward = -(reward -100)
            
            reward *= cfg['reward_multiplier']


            if action==0:
                action_forward+=1
            elif action==1:
                action_rotateLeft+=1
            elif action==2:
                action_rotateRight+=1
            step_count+=1
            total_reward+=reward
            
            # 3. Store performance and training measures
            if end == 1:
                box_collisions += 1
            elif end == 2:
                wall_collisions +=1
            elif end==3:
                max_steps_reached +=1
                reward_list.append(total_reward)
            elif end==4:
                target_reached +=1
                reward_list.append(total_reward)
                
            target_tot=target_reached

            if RESET_UPON_END_SIGNAL[end]:
                break
            else:
                state = next_state

        final_distancetotarget=distancetoTarget
        
        total_approach_to_target= initial_dist_to_target - final_distancetotarget
        print("total approach to target", total_approach_to_target)
        distance_travelled= STEP_SIZE * action_forward
        

        initial_dist_to_targets += initial_dist_to_target
        total_approach_to_targets += total_approach_to_target
        total_distance_travelled += distance_travelled

        values=step_count, wall_collisions, box_collisions, endless_loops, total_reward,agent.eps_threshold, validation_loss, 1, target_reached, max_steps_reached, action_forward,action_rotateLeft,action_rotateRight,distance_travelled,initial_dist_to_target/10,final_distancetotarget/10, total_approach_to_target/10

        validation_list.append(values)
    return validation_list,val_seeds,reward_list, step_count, wall_collisions, box_collisions, endless_loops, total_reward,agent.eps_threshold, validation_loss, 1, target_reached, max_steps_reached, action_forward,action_rotateLeft,action_rotateRight,distance_travelled,initial_dist_to_target,final_distancetotarget, total_approach_to_target,target_tot


def train(agent, environment, img_processing, optimizer, cfg):

    # For reproducability, reset the RNG seed
    torch.manual_seed(cfg['seed'])
    # torch.load()
    # Write header to logfile
    with open(cfg['logfile'], 'w',newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['episode','step_count',
                         'wall_collisions', 'box_collisions',
                         'endless_loops','reward', 'epsilon', 'train_loss', 'validation','target_reached','maxStepReached','forwardStep','leftRotation','rightRotation', 'distanceTravelled', 'initialdistancetoTarget', 'finaldistancetoTarget',   'total_approach_to_target'])
    
    # LOAD THE PREVIOUS MODEL TO CONTINUE TRAINING change best reward according to the minimum possible reward
    # checkpoint= torch.load(r"C:\Users\Berfu\Desktop\Build_Indoor\target_distchange\recent_model_DQN_nr.pth")
    # model.policy_net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # episode= checkpoint['episode']
    # loss_saved=checkpoint['loss']
    # model.policy_net.train()

    best_reward = -1500
    total_loss = 0 # COMMENT OUT TO REGISTER CUMULATIVE LOSS
    wall_collisions = 0
    box_collisions = 0
    episode_reward = 0
    endless_loops = 0
    step_count = 0
    loss_saved=0    
    num_reach=0
    
    for episode in range(cfg['max_episodes']):
        
    # for episode in (episode_range):
        print('episode=', episode)
        with open(cfg['logfile'], 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # Validation loop
        if episode % 50 == 0 or (episode==MAX_EPISODES-1):
            val_performance = validation_loop(agent,environment,img_processing,cfg)

            val_reward_list = val_performance[2]
            targ_total= val_performance[-1]

            print('target reached total',targ_total)
            val_reward=np.average(val_reward_list)
            
            torch.save({'episode' :episode,
            'model_state_dict': agent.policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'last_lr': optimizer.param_groups[0]['lr'],
            'loss': loss_saved}, cfg['recent_model_path'])
            
            
            # Save best model
         
            if val_reward > best_reward:
                print("new best model")
                best_reward = val_reward
                torch.save({'episode' :episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_saved}, cfg['best_model_path'])

                # SAVE THE MODEL WITH MOST TARGET REACHED CONDITIONS

            if targ_total>= num_reach:
                num_reach=targ_total
                torch.save({'episode' :episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_saved}, cfg['t_reached_path'])

            # Write validation performance to log file
            with open(cfg['logfile'], 'a',newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                list_val=val_performance[0]
                for i in range (len(list_val)):
                    row_val=list_val[i]
                    writer.writerow([ episode, *row_val])
                validation_list=[]
                val_reward_list=[]


        # Reset counters
        total_loss = 0 # COMMENT OUT TO REGISTER CUMULATIVE LOSS
        wall_collisions = 0
        box_collisions = 0
        episode_reward = 0
        endless_loops = 0
        step_count = 0
        target_reached=0
        max_steps_reached=0
        action_forward=0
        action_rotateRight=0
        action_rotateLeft=0
        distance_travelled=0
        final_distancetotarget=0
        total_approach_to_target=0
        loss_saved=0
        criterion = nn.SmoothL1Loss()

        # Stop training after (either maximum number of steps or maximum number of episodes)
        if agent.step_count > cfg['max_steps']:
            break

        # Target net is updated once in a few episodes (double Q-learning)
        if episode % cfg['target_update']  == 0:  #episodes
            print('episode {}, target net updated'.format(episode))
            agent.update_target_net()


        # Reset environment at start of episode
        
        seed = torch.randint(250,(1,)).item()
        _, _, _,_ = environment.setRandomSeed(seed)
        
        _, _, initial_dist_to_target,frame_raw = environment.reset(cfg['training_condition'])

        print('initial_dist_to_target', initial_dist_to_target)

        # Create an empty frame stack and fill it with frames
        frame_stack = utils.FrameStack(stack_size=cfg['stack_size'] )
        frame = img_processing(frame_raw).to(agent.device)
        shape=(1,1,128,128)
        frame_dist=torch.ones(shape).to(agent.device) # GENERATE A NEW FRAME WITH DISTANCE TO TARGET INPUT -- COMMENT OUT TO DELETE ADDITIONAL INPUT TO THE NETWORK

        frame_dist[:]=initial_dist_to_target/255 # NORMALIZE WITH THE PIXEL -- COMMENT OUT TO DELETE ADDITIONAL INPUT TO THE NETWORK
        # frame_dist=torch.Tensor(frame_dist / 255.).view(1,1,IMSIZE,IMSIZE)

        for _ in range(cfg['stack_size'] ):
            state = frame_stack.update_with(frame)
        state= torch.cat((state,frame_dist),1) # -- COMMENT OUT TO DELETE ADDITIONAL INPUT TO THE NETWORK

        for t in count():
            # 1. Agent performs a step (based on the current state) and obtains next state
            agent.policy_net.eval()
            action = agent.select_action(state)
            end, reward, distancetoTarget, frame_raw = environment.step(action.item())
            agent_died = cfg['reset_upon_end_signal'][end]
            frame = img_processing(frame_raw).to(agent.device)
            frame_dist[:]=distancetoTarget/255 #-- COMMENT OUT TO DELETE ADDITIONAL INPUT TO THE NETWORK
            next_state = frame_stack.update_with(frame) if not agent_died else None
            next_state= torch.cat((next_state,frame_dist),1) if not agent_died else None #-- COMMENT OUT TO DELETE ADDITIONAL INPUT TO THE NETWORK
            if action==0:
                action_forward+=1
                
            if action==1:
                action_rotateLeft+=1
            if action==2:
                action_rotateRight+=1

            if reward >= 100:
                reward = -(reward -100)
            
            reward *= cfg['reward_multiplier']

            # 3. Push the transition to replay memory (in the right format & shape)
            reward = torch.tensor([reward], device=agent.device,dtype=torch.float)
            action = action.unsqueeze(0)
            agent.memory.push(state, action, next_state, reward)

            # 4. optimize model
            agent.policy_net.train()
            if len(agent.memory) > cfg['replay_start_size']:

                state_action_values, expected_state_action_values = agent.forward()

                # Compute Huber loss
                loss = criterion(state_action_values, expected_state_action_values)                
                total_loss += loss.item()
                loss_saved= total_loss

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                for param in agent.policy_net.parameters(): 
                    param.grad.data.clamp_(-1,1)
                # Update the model parameters
                optimizer.step()

            else:
                # Do not count as optimization loop
                agent.step_count = 0

            # 5. Store performance and training measures
            step_count += 1
            print("step Count=", step_count)
            episode_reward += reward.item()
            
            if end == 1:
                box_collisions += 1
            if end == 2:
                wall_collisions +=1
            if end==3:
                max_steps_reached +=1
            if end==4:
                target_reached +=1
        
             
            # 6. the episode ends here if agent performed any 'lethal' action (specified in RESET_UPON_END_SIGNAL)
            if agent_died:
                break
            else:
                state = next_state
        
        print('avg loss', total_loss/step_count)
            
        final_distancetotarget=distancetoTarget
        
        total_approach_to_target = initial_dist_to_target - final_distancetotarget

        distance_travelled = STEP_SIZE *action_forward  #CHOOSE ONE VERSION AND SWITCH EVERYWHERE
       
            # Write training performance to log file
        with open(cfg['logfile'], 'a',newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([episode,step_count,
                                wall_collisions, box_collisions,
                                endless_loops, episode_reward,agent.eps_threshold,total_loss, 0, target_reached, max_steps_reached, action_forward,action_rotateLeft,action_rotateRight,distance_travelled,initial_dist_to_target/10,final_distancetotarget/10,total_approach_to_target/10]) 

if __name__ == "__main__":
    ### TODO: ARGPARSER ###

    ## Environment
    IMSIZE = 128
    STACK_SIZE = 4
    N_ACTIONS = 3
    STEP_SIZE = 0.2 #appdata.forwardspeed
    IP  = "127.0.0.1" # Ip address that the TCP/IP interface listens to
    PORT = 13000       # Port number that the TCP/IP interface listens to
    environment =  pyClient.Environment(ip = IP, port = PORT, size = IMSIZE) # or choose # DummyEnvironment()

    ## Image processing
    PHOSPHENE_RESOLUTION = 30

    class ImageProcessor(object):
        def __init__(self, phosphene_resolution=None, vision_type=None, imsize=128, canny_threshold=70):
            """ @TODO
            - Extended image processing
            """
            self.thr_high = canny_threshold
            self.thr_low  = canny_threshold // 2
            self.imsize = imsize
            self.vision=vision_type
            if phosphene_resolution is not None:
                self.simulator = utils.PhospheneSimulator(phosphene_resolution=(phosphene_resolution,phosphene_resolution),size=(128,128),
                                                            jitter=0.25,intensity_var=0.9,aperture=.66,sigma=0.60,)
            else:
                self.simulator = None
            

        def __call__(self,state_raw):
            
            if self.vision=='color' or self.vision=='gray' or self.vision=='canny':
                frame = environment.state2usableArray(state_raw)
                frame2=environment.state2usableArray2(state_raw)[...,3:6]
                # print(frame.shape)
                plt.imshow(frame)
                plt.show()
                plt.imshow(frame2)
                plt.show()

            if self.vision=='gray' or self.vision=='canny':
            #     frame= frame.reshape([128,128])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                # print(frame.shape)
                # plt.imshow(frame, cmap='gray')
                # plt.show()
                # plt.imshow(frame2, cmap='gray')
                # plt.show()

            if self.vision=='canny' or self.simulator is not None:
                frame = cv2.Canny(frame, self.thr_low,self.thr_high)
                frame2 = cv2.Canny(frame2, self.thr_low,self.thr_high)
                # plt.imshow(frame, cmap='gray')
                # plt.show()
                # plt.imshow(frame2, cmap='gray')
                # plt.show()
                frame = self.simulator(frame)
                frame2 = self.simulator(frame2)             
                # plt.imshow(frame,cmap='gray')
                # plt.show()
                # plt.imshow(frame2,cmap='gray')
                # plt.show()

            frame = frame.astype('float32')
                # plt.imshow(frame)
                # plt.show()
            
            if self.vision=='gray' or self.vision=='canny':
                return torch.Tensor(frame / 255.).view(1,1,self.imsize, self.imsize)
            if self.vision=='color':
                return torch.Tensor(frame / 255.).view(1,3,self.imsize,self.imsize)

    img_processing = ImageProcessor(phosphene_resolution = PHOSPHENE_RESOLUTION, vision_type='gray')

    # ## DQN Agent
    BATCH_SIZE = 128 
    GAMMA = 0.99
    EPS_START = 0.95
    EPS_END = 0.05
    EPS_DECAY_steps = 4000
    EPS_DECAY = (EPS_START - EPS_END)/EPS_DECAY_steps
    REPLAY_START_SIZE =  1500
    TARGET_UPDATE = 100 #episodes 
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MEMORY_CAPACITY = 12000
    #to make it color make the in_channels=STACK_SIZE*3
    agent = model.DoubleDQNAgent(imsize=IMSIZE,
                     in_channels=STACK_SIZE+1, # DELETE +1 TO DELETE ADDITIONAL INPUT
                     n_actions=N_ACTIONS,
                     memory_capacity=MEMORY_CAPACITY,
                     eps_start=EPS_START,
                     eps_end=EPS_END,
                     eps_delta=EPS_DECAY,
                     gamma_discount = GAMMA,
                     batch_size = BATCH_SIZE,
                     device=DEVICE)

    ## Random Agent

    # BATCH_SIZE = 128 #original 128
    # GAMMA = 0.99 #originally it was 0.5 
    # EPS_START = 0.95
    # EPS_END = 0.05
    # EPS_DECAY_steps = 4000
    # EPS_DECAY = (EPS_START - EPS_END)/EPS_DECAY_steps
    # REPLAY_START_SIZE =  1500
    # TARGET_UPDATE = 10 #episodes
    # DEVICE = torch.device("cuda" if torch.cuda.is_available(1) else "cpu")
    # MEMORY_CAPACITY = 12000
    # agent = model.DoubleDQNAgent(imsize=IMSIZE,
    #                  in_channels=STACK_SIZE,
    #                  n_actions=N_ACTIONS,
    #                  memory_capacity=MEMORY_CAPACITY,
    #                  eps_start=EPS_START,
    #                  eps_end=EPS_END,
    #                  eps_delta=EPS_DECAY,
    #                  gamma_discount = GAMMA,
    #                  batch_size = BATCH_SIZE,
    #                  device=DEVICE)

    ## Optimizer
    LR_DQN = 0.0001
    
    # checkpoint= torch.load(r"C:\Users\Berfu\Desktop\dist_app_posrew1_tr\recent_model_DQN_nr.pth")
    optimizer = optim.Adam(agent.policy_net.parameters(), lr = LR_DQN)
    model=agent

    ## Training parameters
    OUT_PATH = ('/Users/berfukaraca/Desktop/try-im')
    RECENT_MODEL_PATH = os.path.join(OUT_PATH,'recent_model_DQN_nr.pth')
    BEST_MODEL_PATH = os.path.join(OUT_PATH,'best_model_DQN_nr.pth')
    TARGET_PATH = os.path.join(OUT_PATH,'t_reached_path')
    FINAL_MODEL_PATH= os.path.join(OUT_PATH,'final_model_path')
    LOGFILE = os.path.join(OUT_PATH,'train_stats_DQN_nr.csv')
    model_name='Best_model_replay_DQN_nr'
    SEED = 0
    # FOR THE PLAYER FILE -- THERE ARE DIFFERENT CONDITIONS FOR TARGET GETTING
    TRAINING_CONDITION = 3 # the area target generated expands during training
    VALIDATION_CONDITION = 2  #Target generated at a random floor

    MAX_EPISODES =1000 # number of episodes (an episode ends after agent hits a box)
    MAX_STEPS  = 1e6 # number of optimization steps (each time step the model parameters are updated)
    #RESET_AFTER_NR_SIDESTEPS = 5
    RESET_UPON_END_SIGNAL = {0:False,  # Nothing happened
                             1:False,   # Box collision
                             2:False,   # Wall collision
                             3:True, # max steps reached
                             4:True}  # Reached target location
    REWARD_MULTIPLIER        = 1.

    ## Write replay memory to output videos
    def save_replay():
        out = cv2.VideoWriter(os.path.join(OUT_PATH,'{}.avi'.format(model_name)),
                              cv2.VideoWriter_fourcc('M','J','P','G'),
                              2, (IMSIZE,IMSIZE))
        for i, (state, action, next_state, reward) in enumerate(agent.memory.memory):
            frame = (255*state[0,-1].detach().cpu().numpy()).astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.putText(frame,'reward: {:0.1f}'.format(reward.item()),(0,20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.35,color=(0,0,255))
            frame = cv2.putText(frame,'action: {}'.format(action.item()),(0,10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.35,color=(0,0,255))
            out.write(frame)
        out.release()

        

    

    ## Start training
    cfg = dict()
    cfg['recent_model_path']        = RECENT_MODEL_PATH # Save path for model
    cfg['best_model_path']          = BEST_MODEL_PATH # Save path for model
    cfg['logfile']                  = LOGFILE # To save the optimizaiton stats
    cfg['seed']                     = SEED # for reproducability of random factors
    cfg['training_condition']       = TRAINING_CONDITION #THE ACTIONS ARE THE CONDITIONS (TRAINING CONDITION/VALIDATION CONDITION)--> Player file 
    cfg['validation_condition']       = VALIDATION_CONDITION #THE ACTIONS ARE THE CONDITIONS (TRAINING CONDITION/VALIDATION CONDITION)--> Player file 
    cfg['max_episodes']             = MAX_EPISODES
    cfg['max_steps']                = MAX_STEPS # Training stops after either max episodes is reached, or max optimization steps
    cfg['stack_size']               = STACK_SIZE # For frame stacking
    cfg['target_update']            = TARGET_UPDATE #Number of episodes after which DQN target net is updated
    #cfg['reset_after_nr_sidesteps'] = RESET_AFTER_NR_SIDESTEPS # Training is stopped when model keeps side stepping (i.e. it stops reveicing positive rewards)
    cfg['reset_upon_end_signal']    = RESET_UPON_END_SIGNAL # Decide whether to consider different end signals as final state (i.e. box collision, wall collision, step target reached)
    cfg['replay_start_size']        = REPLAY_START_SIZE # Start optimizing when replay memory contains this number of transitions
    cfg['reward_multiplier']        = REWARD_MULTIPLIER # Multiplies the reward signal with this value
    cfg['t_reached_path']           = TARGET_PATH
    cfg['final_model_path']         = FINAL_MODEL_PATH

    print('training...')
    if not os.path.isdir(OUT_PATH):
        os.makedirs(OUT_PATH)
    time_beginning=time.time()
    train(agent, environment, img_processing, optimizer, cfg)
    time_end=time.time()
    print("training duration",time_end-time_beginning)
    print('finished training')

    save_replay()
