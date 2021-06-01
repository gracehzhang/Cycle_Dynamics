

import numpy as np
import torch
import gym
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


from tqdm import tqdm
from PIL import Image
import moviepy.editor as mpy
from env_utils import SawyerECWrapper

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
    def __init__(self,policy_path,state_dim,action_dim,max_action,opt):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.opt = opt
        self.weight_path = policy_path
        self.actor.load_state_dict(torch.load(self.weight_path))
        print('policy weight loaded!')
        self.stack_agent = Stackimg(opt)
        self.env_logs1 = os.path.join(self.opt.log_root, '{}_data'.format(self.opt.source_env))
        self.env_logs2 = os.path.join(self.opt.log_root, '{}_data'.format(self.opt.env))
        self.clip_range = 5
        self.mean1,self.std1 = self.get_mean_std(opt.data_type1,opt.data_id1,self.env_logs1)
        self.mean2,self.std2 = self.get_mean_std(opt.data_type2,opt.data_id2,self.env_logs2)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def select_cross_action(self,img,gxmodel,axmodel):
        img = self.stack_agent.push(img[::-1, :, :])
        output = gxmodel(img.unsqueeze(0)).squeeze()
        # state = gxmodel(state.unsqueeze(0))
        state = self.mean1.clone()
        state[:output.shape[0]] = output
        state = state * self.std1 + self.mean1
        state = state.cpu().data.numpy()
        action = self.select_action(state)
        action = axmodel(torch.tensor(action).float().cuda().unsqueeze(0)).squeeze()
        action = action.cpu().data.numpy()
        return action

    def get_mean_std(self,data_type,data_id,env_logs):
        data_path = os.path.join(env_logs, '{}_{}'.format(data_type, data_id))
        mean_std_path = os.path.join(data_path,'now_state.npy')
        data = np.load(mean_std_path)
        mean = torch.tensor(data.mean(0)).float().cuda()
        std = torch.tensor(data.std(0)).float().cuda()
        std[(abs((std<0.1).type(torch.uint8)))] = 1
        return mean,std


class CrossImgPolicy:
    def __init__(self,opt):
        self.opt = opt
        self.env_name = opt.env
        self.policy_path = os.path.join(opt.log_root,
                 '{}_base/models/TD3_{}_0_actor'.format(opt.env, opt.env))
        self.state_dim = opt.state_dim1
        self.action_dim = opt.action_dim1
        self.max_action = 1
        print(self.env_name, self.state_dim, self.action_dim)
        self.policy = TD3(self.policy_path,
                          self.state_dim,
                          self.action_dim,
                          self.max_action,
                          self.opt)
        self.env = gym.make(self.env_name)
        if "SawyerPush" in self.opt.env:
            self.env = SawyerECWrapper(self.env, opt.env)
            self.env._max_episode_steps = 70
        self.env.seed(100)

    def eval_policy(self,
                    iter,
                    gxmodel=None,
                    axmodel=None,
                    imgpath=None,
                    eval_episodes=10):
        eval_env = self.env
        state_buffer = []
        action_buffer = []
        avg_reward,new_reward = 0.,0.
        success_rate = 0.
        save_flag = False
        if imgpath is not None:
            if not os.path.exists(imgpath):
                os.mkdir(imgpath)
            save_flag = True

        for i in tqdm(range(eval_episodes)):
            state, done = eval_env.reset(), False
            if save_flag:
                episode_path = os.path.join(imgpath,'iteration_{}_episode_{}.mp4'.format(iter, i))
                frames = []
            count = 0
            while not done:
                state = np.array(flatten_state(state))
                img, depth = self.env.sim.render(mode='offscreen', width=100, height=100, depth=True)
                with torch.no_grad():
                    action = self.policy.select_cross_action(img,gxmodel,axmodel)
                state_buffer.append(state)
                action_buffer.append(action)
                state, reward, done, info = eval_env.step(action)
                state = flatten_state(state)
                if ("first_success" in info.keys() and info["first_success"] == 1):
                    success_rate += 1
                elif ("episode_success" in info.keys() and info["episode_success"] == True):
                    success_rate += 1
                avg_reward += reward

                if save_flag:
                    img = eval_env.sim.render(mode='offscreen', camera_name='track', width=500, height=500)
                    frames.append(img[::-1, :, :])
                count += 1
            if save_flag:
                self._save_video(episode_path, frames)
                if i >= 3:
                    save_flag = False
        avg_reward /= eval_episodes
        success_rate /= eval_episodes

        print("-----------------------------------------------")
        print("Evaluation over {} episodes: {:.3f}, {:.3f}".format(eval_episodes, avg_reward, success_rate))
        print("-----------------------------------------------")

        return avg_reward, success_rate

    def _save_video(self, fname, frames, fps=15.0):
        """ Saves @frames into a video with file name @fname. """

        def f(t):
            frame_length = len(frames)
            new_fps = 1.0 / (1.0 / fps + 1.0 / frame_length)
            idx = min(int(t * new_fps), frame_length - 1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames) / fps + 2)

        video.write_videofile(fname, fps, verbose=False)


class Stackimg:
    def __init__(self,opt):
        self.stack_n = opt.stack_n
        self.img_size = opt.img_size
        self.env = opt.env
        self.trans_stack = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.buffer = []

    def push(self,img):
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img = transforms.ToPILImage()(img)
        img = self.trans_stack(img)

        if len(self.buffer)<self.stack_n:
            self.buffer = [img]*self.stack_n
        else:
            self.buffer = self.buffer[1:]
            self.buffer.append(img)
        return torch.stack(self.buffer,0).float().cuda()

