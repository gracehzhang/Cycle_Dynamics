import numpy as np
import torch
import gym
import argparse
import os
import copy
#import sys
#sys.path.insert(1, '/home/grace/sim2real')

import utils
import TD3
# from s2r_method.s2r_utils.env import make_s2r_env
from env_utils import SawyerECWrapper

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # for running on macOS


def safe_path(path):
        if not os.path.exists(path):
                os.mkdir(path)
        return path

def flatten_state(state):
    if isinstance(state, dict):
        state_cat = []
        for k,v in state.items():
            state_cat.extend(v)
        state = state_cat
    return state

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
        eval_env = gym.make(env_name)
        if "SawyerPush" in args.env:
            eval_env = SawyerECWrapper(eval_env, args.env)
            eval_env._max_episode_steps = 70
        eval_env.seed(seed + 100)

        avg_reward = 0.
        success_rate = 0.
        for _ in range(eval_episodes):
                state, done = eval_env.reset(), False
                state = flatten_state(state)
                while not done:
                        action = policy.select_action(np.array(flatten_state(state)))
                        state, reward, done, info = eval_env.step(action)
                        if ("first_success" in info.keys() and info["first_success"]):
                            success_rate += 1
                        avg_reward += reward

        avg_reward /= eval_episodes
        success_rate /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, {success_rate:.3f}")
        print("---------------------------------------")
        return avg_reward

def main(args):
        file_name = f"{args.policy}_{args.env}_{args.seed}"
        print("---------------------------------------")
        print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")

        log_path = safe_path(os.path.join(args.log_root, '{}_base'.format(args.env)))
        result_path = safe_path(os.path.join(log_path, 'results'))
        model_path = safe_path(os.path.join(log_path, 'models'))
        
        '''
        ### s2r hacks
        s2r_parser = argparse.ArgumentParser()
        s2r_parser.add_argument("--encoder_type", default="mlp")
        s2r_parser.add_argument("--end_effector", default=True)
        s2r_parser.add_argument("--screen_width", type=int, default=480)
        s2r_parser.add_argument("--screen_height", type=int, default=480)
        s2r_parser.add_argument("--action_repeat", type=int, default=1)
        s2r_parser.add_argument("--puck_friction", type=float, default=2.0)
        s2r_parser.add_argument("--puck_mass", type=float, default=0.01)
        s2r_parser.add_argument("--unity",  default=False)
        s2r_parser.add_argument("--unity_editor", default=False)
        s2r_parser.add_argument("--virtual_display",  default=None)
        s2r_parser.add_argument("--port", default=1050)
        s2r_parser.add_argument("--absorbing_state", default=False)
        s2r_parser.add_argument("--dr", default=False)
        s2r_parser.add_argument("--env", default=None)
        s2r_args = s2r_parser.parse_args()
        import ipdb;ipdb.set_trace()
        env = make_s2r_env(args.env, s2r_args, env_type="real")
        '''
        env = gym.make(args.env)
        if "SawyerPush" in args.env:
            env = SawyerECWrapper(env, args.env)
            env._max_episode_steps = 70
        # Set seeds
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        try:
            state_dim = env.observation_space.shape[0]
        except:
            state_dim = 16 #env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "discount": args.discount,
                "tau": args.tau,
        }

        # Initialize policy
        if args.policy == "TD3":
                # Target policy smoothing is scaled wrt the action scale
                kwargs["policy_noise"] = args.policy_noise * max_action
                kwargs["noise_clip"] = args.noise_clip * max_action
                kwargs["policy_freq"] = args.policy_freq
                policy = TD3.TD3(**kwargs)

        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

        # Evaluate untrained policy
        evaluations = [eval_policy(policy, args.env, args.seed)]

        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        success = False
        reach_reward = 0
        push_reward = 0
        cylinder_to_target = 100
        for t in range(int(args.max_timesteps)):
                state = flatten_state(state)
                episode_timesteps += 1

                # Select action randomly or according to policy
                if t < args.start_timesteps:
                        action = env.action_space.sample()
                else:
                        action = (
                                        policy.select_action(np.array(state))
                                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                        ).clip(-max_action, max_action)

                # Perform action
                next_state, reward, done, info = env.step(action)
                next_state = flatten_state(next_state)
                done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

                if ("first_success" in info.keys() and info["first_success"]):
                    success = True

                # reach_reward += info["reward_reach"]
                # push_reward += info["reward_push"]
                # cylinder_to_target = min(cylinder_to_target, info["cylinder_to_target"])

                # Store data in replay buffer
                replay_buffer.add(state, action, next_state, reward, done_bool)

                state = next_state
                episode_reward += reward

                # Train agent after collecting sufficient data
                if t >= args.start_timesteps:
                        policy.train(replay_buffer, args.batch_size)

                if done:
                        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                        # reach_reward /= episode_timesteps
                        # push_reward /= episode_timesteps
                        #  Reach Reward: {reach_reward:.3f} Push Reward: {push_reward:.3f} cylinder_to_target: {cylinder_to_target:.3f}
                        print(
                                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Success: {success}")
                        # Reset environment
                        success = False
                        state, done = env.reset(), False
                        episode_reward = 0
                        reach_reward, push_reward = 0, 0
                        cylinder_to_target = 100
                        episode_timesteps = 0
                        episode_num += 1

                # Evaluate episode
                if (t + 1) % args.eval_freq == 0:
                        evaluations.append(eval_policy(policy, args.env, args.seed))
                        np.save(os.path.join(result_path, '{}'.format(file_name)), evaluations)
                        if args.save_model: policy.save(os.path.join(model_path, '{}'.format(file_name)))



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
        parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
        parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
        parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
        parser.add_argument("--max_timesteps", default=1.7e5, type=int)   # Max time steps to run environment
        parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
        parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
        parser.add_argument("--discount", default=0.99)                 # Discount factor
        parser.add_argument("--tau", default=0.005)                     # Target network update rate
        parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
        parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
        parser.add_argument("--policy_freq", default=1, type=int)       # Frequency of delayed policy updates
        parser.add_argument("--save_model", default=True)               # Save model and optimizer parameters
        parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

        parser.add_argument("--log_root", default="../../../logs/cross_morphology")
        args = parser.parse_args()
        
        main(args)
