
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from options import get_options
from cycle.data import IterCycleData
from cycle.dyncycle import CycleGANModel
from cycle.utils import init_logs

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # for running on macOS


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def flatten_state(state):
    if isinstance(state, dict):
        state_cat = []
        for k,v in state.items():
            state_cat.extend(v)
        state = state_cat
    return state
# setup_seed(0)


def add_errors(model,display):
    errors = model.get_current_errors()
    for key, value in errors.items():
        if key=='G_act_B':
            display += '\n'
        display += '{}:{:.4f}  '.format(key, value)
    return display

def iter_train(opt):
    logs = init_logs(opt)
    opt.istrain = False
    eval_logs = init_logs(opt)
    opt.istrain = True

    data_agent = IterCycleData(opt)
    setup_seed(args.seed)
    model = CycleGANModel(opt)
    model.update(opt)
    best_reward = -1000
    for iter in range(opt.iterations):
        # Train
        opt.istrain = True
        best_reward = train(opt, data_agent, model, iter, best_reward, logs)
        if opt.finetune:
            opt.pair_n = 700
            opt.display_gap = 100
            opt.eval_gap = 100

        # Test
        opt.istrain = False
        opt.init_start = False
        with torch.no_grad():
            test(opt, iter, eval_logs)

        # Collect Data

        collect_data(opt, data_agent, model)


def collect_data(opt, data_agent, model):
    ### Source Dataset
    policy_path = os.path.join(opt.log_root,
                               '{}_base/models/TD3_{}_0_actor'.format(opt.env, opt.env))
    data_agent.create_data(opt.source_env, 1, opt.episode_n, policy=policy_path)

    ### Target Dataset
    data_agent.create_data(opt.env, 2, opt.episode_n, img=True, model=model)


def train(args, data_agent, model, iter, best_reward, logs):
    txt_logs, img_logs, weight_logs = logs
    model.fengine.train_statef(data_agent.data1)
    model.cross_policy.eval_policy(
                        iter,
                        gxmodel=model.netG_B,
                        axmodel=model.net_action_G_A,
                        eval_episodes=1)

    end_id = 0
    for iteration in range(3):

        args.lr_Gx = 1e-4
        args.lr_Ax = 0
        model.update(args)

        start_id = end_id
        end_id = start_id + args.pair_n
        for batch_id in range(start_id,end_id):
            # print(iteration, batch_id)
            item = data_agent.get_img_sample()
            data1,data2 = item
            model.set_input(item)
            model.optimize_parameters()
            real,fake = model.fetch()

            if (batch_id + 1) % args.display_gap == 0:
                display = '\n===> Batch[{}/{}]'.format(batch_id+1, args.pair_n)
                print(display)
                display = add_errors(model, display)
                txt_logs.write('{}\n'.format(display))
                txt_logs.flush()

                path = os.path.join(img_logs, 'imgA_{}.jpg'.format(batch_id + 1))
                model.visual(path)

            if (batch_id + 1) % args.eval_gap == 0:
                reward, success_rate=  model.cross_policy.eval_policy(
                    iter,
                    gxmodel=model.netG_B,
                    axmodel=model.net_action_G_A,
                    eval_episodes=args.eval_n)
                if reward>best_reward:
                    best_reward = reward
                    model.save(weight_logs)
                print('iter:{:.1f}  best_reward:{:.1f}  cur_reward:{:.1f} cur_success:{:.1f}'.format(iter, best_reward,reward,success_rate))
                txt_logs.write('iter:{:.1f}  best_reward:{:.1f}  cur_reward:{:.1f} cur_succes_rate:{:.1f}\n'.format(iter, best_reward, reward, success_rate))
                txt_logs.flush()

        args.init_start = False
        args.lr_Gx = 0
        args.lr_Ax = 1e-4
        model.update(args)

        start_id = end_id
        end_id = start_id + args.pair_n
        for batch_id in range(start_id,end_id):
            item = data_agent.get_img_sample()
            data1, data2 = item
            model.set_input(item)
            model.optimize_parameters()
            real, fake = model.fetch()

            if (batch_id + 1) % args.display_gap == 0:
                display = '\n===> Batch[{}/{}]'.format(batch_id+1, args.pair_n)
                print(display)
                display = add_errors(model, display)
                txt_logs.write('{}\n'.format(display))
                txt_logs.flush()

                path = os.path.join(img_logs, 'imgA_{}.jpg'.format(batch_id + 1))
                model.visual(path)

            if (batch_id + 1) % args.eval_gap == 0:
                reward, success_rate = model.cross_policy.eval_policy(
                    iter,
                    gxmodel=model.netG_B,
                    axmodel=model.net_action_G_A,
                    eval_episodes=args.eval_n)
                if reward>best_reward:
                    best_reward = reward
                    model.save(weight_logs)

                print('iter:{:.1f}  best_reward:{:.1f}  cur_reward:{:.1f} cur_success:{:.1f}'.format(iter, best_reward,reward,success_rate))
                txt_logs.write('iter:{:.1f}  best_reward:{:.1f}  cur_reward:{:.1f} cur_success:{:.1f}\n'.format(iter, best_reward, reward, success_rate))
                txt_logs.flush()

    return best_reward


def test(args, iter, logs):
    txt_logs, img_logs, weight_logs = logs
    # # data_agent = CycleData(args)
    model = CycleGANModel(args)
    # model.fengine.train_statef(data_agent.data1)
    print(weight_logs)
    model.load(weight_logs)
    model.update(args)

    reward, success_rate = model.cross_policy.eval_policy(
        iter,
        gxmodel=model.netG_B,
        axmodel=model.net_action_G_A,
        imgpath=img_logs,
        eval_episodes=1000)

    txt_logs.write('Iteration: {}, Final Evaluation: {}, Success Rate: {}\n'.format(iter, reward, success_rate))
    txt_logs.flush()

if __name__ == '__main__':
    args = get_options()

    iter_train(args)




