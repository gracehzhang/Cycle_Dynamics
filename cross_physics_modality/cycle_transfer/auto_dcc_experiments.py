import random
import os
from absl import logging, flags, app
from multiprocessing import Queue, Manager
from pathos import multiprocessing
import traceback
import time

which_gpus = [1]
max_worker_num = 3
envs = ["gym_mod:GymFetchReach-v0"] #["gym_mod:GymHalfCheetahDM-v0","gym_mod:GymWalkerDM-v0"] #,"gym_mod:GymInvertedPendulumDM-v0"]
state_dims = [16] #[17,17] #,4]
action_dims = [4] #[6,6] #,1]
seeds = [0,1,2,3,4]
data_id = 1
data_type = "easy"

def _init_device_queue(max_worker_num):
    m = Manager()
    device_queue = m.Queue()
    for i in range(max_worker_num):
        idx = i % len(which_gpus)
        gpu = which_gpus[idx]
        device_queue.put(gpu)
    return device_queue

def run():
    """Run trainings with all possible parameter combinations in
    the configured space.
    """

    process_pool = multiprocessing.Pool(
        processes=max_worker_num, maxtasksperchild=1)
    device_queue = _init_device_queue(max_worker_num)

    for seed in seeds:
        for env, state_dim, action_dim in zip(envs, state_dims, action_dims):
            command = "python alignexp.py --eval_n 10 --data_type1 'base' --data_id1 {} --data_type2 {} --data_id2 {}  --seed {} --env {} --state_dim1={} --action_dim1={} --state_dim2={} --action_dim2={}".format(data_id, data_type, data_id, seed, env, state_dim, action_dim, state_dim, action_dim)
            print(command)
            process_pool.apply_async(
                func=_worker,
                args=[command, device_queue],
                error_callback=lambda e: logging.error(e))

    process_pool.close()
    process_pool.join()

def _worker(command, device_queue):
    # sleep for random seconds to avoid crowded launching
    try:
        time.sleep(random.uniform(0, 10))

        device = device_queue.get()

        logging.set_verbosity(logging.INFO)

        logging.info("command %s" % command)
        os.system("CUDA_VISIBLE_DEVICES=%d " % device + command)

        device_queue.put(device)
    except Exception as e:
        logging.info(traceback.format_exc())
        raise e
run()
