
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gym
import numpy as np
from PIL import Image
from torchvision import transforms


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
        self.env_logs1 = os.path.join(self.log_root, '{}_data'.format(self.opt.source_env))
        self.env_logs2 = os.path.join(self.log_root, '{}_data'.format(self.opt.env))
        self.data_root1 = os.path.join(self.env_logs1, '{}_{}'.format(self.opt.data_type1, self.opt.data_id1))
        self.data_root2 = os.path.join(self.env_logs2, '{}_{}'.format(self.opt.data_type2, self.opt.data_id2))

        self.stack_n = opt.stack_n
        self.img_size = opt.img_size
        self.trans_stack = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.data1 = self.collect(self.data_root1, opt.state_dim1)
        self.data2 = self.collect(self.data_root2, opt.state_dim2, img=True)
        self.finetune = opt.finetune
        print('----------- Dataset initialized ---------------')
        print('-----------------------------------------------\n')
        self.sample_n1 = self.data1[1].shape[0]
        self.sample_n2 = self.data2[1].shape[0]
        print(self.data_root1, self.data_root2)
        print(self.sample_n1, self.sample_n2)

    # def sample(self, batch_size=32):
    #     id1 = random.sample(range(self.sample_n1), batch_size)
    #     sample1 = (self.data1[0][id1], self.data1[1][id1], self.data1[2][id1])
    #     id2 = random.sample(range(self.sample_n2), batch_size)
    #     sample2 = (self.data2[0][id2], self.data2[1][id2], self.data2[2][id2])
    #     return sample1, sample2

    def read_img(self,path):
        img = Image.open(path)
        img = transforms.ToTensor()(img)
        img = transforms.ToPILImage()(img)
        img = self.trans_stack(img)
        return img

    def get_state_sample(self, batch_size=32):
        ids = random.sample(range(self.sample_n1), batch_size)
        sample = [self.data1[0][ids], self.data1[1][ids], self.data1[2][ids]]
        return sample

    def get_img_sample(self, batch_size=32):
        sample1 = self.get_state_sample(batch_size)

        id2 = random.sample(range(self.sample_n2-self.stack_n+1), batch_size)
        img_now,img_nxt = [],[]
        for i in id2:
            img_now.append(torch.stack(self.data2[0][i:i+self.stack_n]))
            img_nxt.append(torch.stack(self.data2[2][i:i+self.stack_n]))
        img_now = torch.stack(img_now,0).numpy()
        img_nxt = torch.stack(img_nxt,0).numpy()

        sample2 = [img_now, self.data2[1][id2], img_nxt]

        return sample1, sample2

    def collect(self, data_folder,state_dim,img=False):
        if img:
            img_path = os.path.join(data_folder, 'imgs')
            now_obs, nxt_obs = self.get_imgs(img_path)
            mean = None
            std = None
        else:
            now_path = os.path.join(data_folder, 'now_state.npy')
            nxt_path = os.path.join(data_folder, 'next_state.npy')
            now_obs = np.load(now_path)[:, :state_dim]
            nxt_obs = np.load(nxt_path)[:, :state_dim]

            mean = now_obs.mean(0)
            std = now_obs.std(0)
            std[(abs(std) < 0.1)] = 1
            now_obs = (now_obs - mean) / std
            nxt_obs = (nxt_obs - mean) / std

        act_path = os.path.join(data_folder, 'action.npy')
        action = np.load(act_path)
        
        return [now_obs, action, nxt_obs, mean, std]

    def get_imgs(self,img_path):
        episode_list = os.listdir(img_path)
        episode_list = sorted(episode_list,key=lambda x:int(x.split('-')[1]))
        now_img, nxt_img = [], []
        for dir in episode_list:
            episode_path = os.path.join(img_path,dir)
            tmp = os.listdir(episode_path)
            tmp = sorted(tmp,key=lambda x:int(x.split('_')[-1].split('.')[0]))
            tmp = [os.path.join(episode_path,x) for x in tmp]
            
            imglist = []
            for imgpath in tmp:
                imglist.append(self.read_img(imgpath))
            now_img.extend(imglist[:-1])
            nxt_img.extend(imglist[1:])
        
        return now_img, nxt_img

    def create_data(self, env, data_i, episode_n, img=False, model=None, policy=None):
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
            if img:
                img_obs = env.sim.render(mode='offscreen', camera_name='track', width=100, height=100, depth=False)
                self.add_observation(img_obs, img)
            else:
                self.add_observation(observation, img)
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
                    if img:
                        action = model.sample_action(img_obs)
                    else:
                        action = model.sample_action(observation)
                observation, reward, done, info = env.step(action)
                observation = flatten_state(observation)
                if img:
                    img_obs = env.sim.render(mode='offscreen', camera_name='track', width=100, height=100, depth=False)
                    self.add_observation(img_obs, img)
                else:
                    self.add_observation(observation, img)

                self.add_action(action)

                # path = os.path.join(episode_path, 'img_{}_{}.jpg'.format(i_episode, t + 1))
                # self.check_and_save(path)
                t += 1

                if done:
                    print("Episode {} finished after {} timesteps".format(i_episode,t))
                    total_samples += t
                    break
            self.merge_buffer(img)
        env.close()
        self.collect_data(data_i, img)
        print("{} total samples collected, dataset size: {}, {}".format(total_samples, self.sample_n1, self.sample_n2))

    def collect_data(self, data_i, img):
        #if (data_i == 2):
        #    import ipdb; ipdb.set_trace()
        self.norm_state(img)
        self.pair_n = len(self.now_obs)
        assert (self.pair_n == len(self.next_obs))
        assert (self.pair_n == len(self.action))

        if data_i == 1:
            mean, std = self.data1[-2:]
            # state_dim = self.opt.state_dim1
        elif data_i == 2:
            mean, std = self.data2[-2:]
            # state_dim = self.opt.state_dim2

        now_obs = self.now_obs #[:, :state_dim]
        nxt_obs = self.next_obs #[:, :state_dim]
        action = self.action

        if not img:
            now_obs = (now_obs-mean)/std
            nxt_obs = (nxt_obs-mean)/std

        if data_i == 1:
            if self.finetune:
                self.data1[0] = now_obs
                self.data1[1] = action
                self.data1[2] = nxt_obs
            else:
                self.data1[0] = np.concatenate((self.data1[0], now_obs))
                self.data1[1] = np.concatenate((self.data1[1], action))
                self.data1[2] = np.concatenate((self.data1[2], nxt_obs))
            self.sample_n1 = self.data1[1].shape[0]
        elif data_i == 2:
            if self.finetune:
                self.data2[0] = now_obs
                self.data2[1] = action
                self.data2[2] = nxt_obs
            else:
                self.data2[0].extend(now_obs) # = np.concatenate((self.data2[0], now_obs))
                self.data2[1] = np.concatenate((self.data2[1], action))
                self.data2[2].extend(nxt_obs) # = np.concatenate((self.data2[2], nxt_obs))
            self.sample_n2 = self.data2[1].shape[0]

        return


    def norm_state(self, img):
        if not img:
            self.now_obs = np.vstack(self.now_obs)
            self.next_obs = np.vstack(self.next_obs)
        self.action = np.vstack(self.action)

    def reset_buffer(self):
        self.joint_pose_buffer = []
        self.achieved_goal_buffer = []
        self.goal_pos_buffer = []
        self.action_buffer = []
        self.imgs = []

        self.now_obs = []
        self.next_obs = []
        self.action = []

    def add_image(self, img):
        img = Image.fromarray(img[::-1, :, :])
        img = transforms.ToTensor()(img)
        img = transforms.ToPILImage()(img)
        img = self.trans_stack(img) #.unsqueeze(0).numpy()
        self.imgs.append(img)

    def add_observation(self,observation,img):
        if img:
            self.add_image(observation)
        else:
            self.joint_pose_buffer.append(observation)

    def add_action(self,action):
        self.action_buffer.append(action)

    def merge_buffer(self,img):
        if img:
            self.now_obs += self.imgs[:-1]
            self.next_obs += self.imgs[1:]
            self.action += self.action_buffer #[:-1]
            if (len(self.now_obs) != len(self.action)):
                import ipdb; ipdb.set_trace()
        else:
            self.now_obs += self.joint_pose_buffer[:-1]
            self.next_obs += self.joint_pose_buffer[1:]
            self.action += self.action_buffer

        self.joint_pose_buffer = []
        self.achieved_goal_buffer = []
        self.goal_pos_buffer = []
        self.action_buffer = []
        self.imgs = []



