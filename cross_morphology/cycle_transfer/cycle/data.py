
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gym
import numpy as np
from PIL import Image

def flatten_state(state):
    if isinstance(state, dict):
        state_cat = []
        for k,v in state.items():
            state_cat.extend(v)
        state = state_cat
    return state

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class TD3(object):
    def __init__(self,policy_path,state_dim,action_dim,max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.weight_path = policy_path
        self.actor.load_state_dict(torch.load(self.weight_path))
        print('policy weight loaded!')

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def online_action(self,state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = self.axmodel(state,self.actor(state)).cpu().data.numpy().flatten()
        return action

    def online_axmodel(self,state,axmodel):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = axmodel(state,self.actor(state)).cpu().data.numpy().flatten()
        return action

class CycleData:
    def __init__(self, opt):
        self.opt = opt
        self.episode_n = opt.episode_n
        self.log_root = opt.log_root
        self.env_logs = os.path.join(self.log_root, '{}_data'.format(self.opt.env))
        self.data_root1 = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type1, self.opt.data_id1))
        self.data_root2 = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type2, self.opt.data_id2))
        self.data1 = self.collect(self.data_root1, opt.state_dim1)
        self.data2 = self.collect(self.data_root2, opt.state_dim2)
        print('----------- Dataset initialized ---------------')
        print('-----------------------------------------------\n')
        self.sample_n1 = self.data1[0].shape[0]
        self.sample_n2 = self.data2[0].shape[0]
        print(self.sample_n1, self.sample_n2)

    def sample(self, batch_size=32):
        id1 = random.sample(range(self.sample_n1), batch_size)
        sample1 = (self.data1[0][id1], self.data1[1][id1], self.data1[2][id1])
        id2 = random.sample(range(self.sample_n2), batch_size)
        sample2 = (self.data2[0][id2], self.data2[1][id2], self.data2[2][id2])
        return sample1, sample2

    def collect(self, data_folder,state_dim):
        now_path = os.path.join(data_folder, 'now_state.npy')
        nxt_path = os.path.join(data_folder, 'next_state.npy')
        act_path = os.path.join(data_folder, 'action.npy')
        now_obs = np.load(now_path)[:,:state_dim]
        nxt_obs = np.load(nxt_path)[:,:state_dim]
        action = np.load(act_path)

        mean = now_obs.mean(0)
        std = now_obs.std(0)
        std[(abs(std) < 0.1)] = 1
        now_obs = (now_obs-mean)/std
        nxt_obs = (nxt_obs-mean)/std

        return (now_obs, action, nxt_obs)


class IterCycleData:
    def __init__(self, opt):
        self.opt = opt
        self.episode_n = opt.episode_n
        self.log_root = opt.log_root
        self.env_logs = os.path.join(self.log_root, '{}_data'.format(self.opt.env))
        self.data_root1 = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type1, self.opt.data_id1))
        self.data_root2 = os.path.join(self.env_logs, '{}_{}'.format(self.opt.data_type2, self.opt.data_id2))
        self.data1 = self.collect(self.data_root1, opt.state_dim1)
        self.data2 = self.collect(self.data_root2, opt.state_dim2)
        print('----------- Dataset initialized ---------------')
        print('-----------------------------------------------\n')
        self.sample_n1 = self.data1[0].shape[0]
        self.sample_n2 = self.data2[0].shape[0]
        print(self.sample_n1, self.sample_n2)

    def sample(self, batch_size=32):
        id1 = random.sample(range(self.sample_n1), batch_size)
        sample1 = (self.data1[0][id1], self.data1[1][id1], self.data1[2][id1])
        id2 = random.sample(range(self.sample_n2), batch_size)
        sample2 = (self.data2[0][id2], self.data2[1][id2], self.data2[2][id2])
        return sample1, sample2

    def collect(self, data_folder,state_dim):
        now_path = os.path.join(data_folder, 'now_state.npy')
        nxt_path = os.path.join(data_folder, 'next_state.npy')
        act_path = os.path.join(data_folder, 'action.npy')
        now_obs = np.load(now_path)[:,:state_dim]
        nxt_obs = np.load(nxt_path)[:,:state_dim]
        action = np.load(act_path)

        mean = now_obs.mean(0)
        std = now_obs.std(0)
        std[(abs(std) < 0.1)] = 1
        now_obs = (now_obs-mean)/std
        nxt_obs = (nxt_obs-mean)/std

        return [now_obs, action, nxt_obs, mean, std]

    def create_data(self, env, data_i, episode_n, model=None, policy=None):
        env = gym.make(env)
        env.seed(0)
        random.seed(0)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        model = model
        if policy is not None:
            policy = TD3(policy, state_dim, action_dim, max_action)

        self.reset_buffer()
        total_samples = 0
        i_episode = 0
        while total_samples < episode_n:
            observation, done, t = env.reset(), False, 0
            observation = flatten_state(observation)
            self.add_observation(observation)
            # episode_path = os.path.join(self.img_path,'episode-{}'.format(i_episode))
            # if not os.path.exists(episode_path):
            #     os.mkdir(episode_path)
            # path = os.path.join(episode_path, 'img_{}_{}.jpg'.format(i_episode, 0))
            # self.check_and_save(path)
            i_episode += 1
            while not done:
                if policy is not None:
                    action = policy.select_action(observation)
                elif model is not None:
                    action = model.sample_action(observation)
                observation, reward, done, info = env.step(action)
                observation = flatten_state(observation)
                self.add_action(action)
                self.add_observation(observation)

                # path = os.path.join(episode_path, 'img_{}_{}.jpg'.format(i_episode, t + 1))
                # self.check_and_save(path)
                t += 1

                if done:
                    print("Episode {} finished after {} timesteps".format(i_episode,t))
                    total_samples += t
                    break
            self.merge_buffer()

        env.close()
        self.collect_data(data_i, state_dim)
        print("{} total samples collected, dataset size: {}, {}".format(total_samples, self.sample_n1, self.sample_n2))

    def check_and_save(self,path):
        img = self.env.sim.render(mode='offscreen', camera_name='track', width=256, height=256, depth=False)
        img = Image.fromarray(img[::-1, :, :])
        img.save(path)

    def collect_data(self, data_i, state_dim):
        self.norm_state()
        self.pair_n = self.now_state.shape[0]
        assert (self.pair_n == self.next_state.shape[0])
        assert (self.pair_n == self.action.shape[0])

        if data_i == 1:
            mean, std = self.data1[-2:]
            state_dim = self.opt.state_dim1
        elif data_i == 2:
            mean, std = self.data2[-2:]
            state_dim = self.opt.state_dim2

        now_obs = self.now_state[:,:state_dim]
        nxt_obs = self.next_state[:,:state_dim]
        action = self.action

        now_obs = (now_obs-mean)/std
        nxt_obs = (nxt_obs-mean)/std

        if data_i == 1:
            self.data1[0] = np.concatenate((self.data1[0], now_obs))
            self.data1[1] = np.concatenate((self.data1[1], action))
            self.data1[2] = np.concatenate((self.data1[2], nxt_obs))
            self.sample_n1 = self.data1[0].shape[0]
        elif data_i == 2:
            self.data2[0] = np.concatenate((self.data2[0], now_obs))
            self.data2[1] = np.concatenate((self.data2[1], action))
            self.data2[2] = np.concatenate((self.data2[2], nxt_obs))
            self.sample_n2 = self.data2[0].shape[0]

        return

    def norm_state(self):
        self.now_state = np.vstack(self.now_state)
        self.next_state = np.vstack(self.next_state)
        self.action = np.vstack(self.action)

    def reset_buffer(self):
        self.joint_pose_buffer = []
        self.achieved_goal_buffer = []
        self.goal_pos_buffer = []
        self.action_buffer = []

        self.now_state = []
        self.next_state = []
        self.action = []

    def add_observation(self,observation):
        self.joint_pose_buffer.append(observation)

    def add_action(self,action):
        self.action_buffer.append(action)

    def merge_buffer(self):
        self.now_state += self.joint_pose_buffer[:-1]
        self.next_state += self.joint_pose_buffer[1:]
        self.action += self.action_buffer

        self.joint_pose_buffer = []
        self.achieved_goal_buffer = []
        self.goal_pos_buffer = []
        self.action_buffer = []



