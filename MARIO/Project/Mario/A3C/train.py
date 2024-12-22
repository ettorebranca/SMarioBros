import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import ActorCritic
from src.process import local_train, local_test
from src.glob_optim import GlobalAdam as optim
import torch.multiprocessing as _mp
import shutil

def get_args():
    parser = argparse.ArgumentParser("""Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning """)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")   
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=0.95, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.1, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)    ### number of steps between communication with global model and updatings of its weights
    parser.add_argument("--num_global_steps", type=int, default=100000)  ## max steps for each single process
    parser.add_argument("--num_processes", type=int, default=6)   
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings") 
    parser.add_argument("--max_actions", type=int, default=500, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="A3C_trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False, help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=False)  
    args = parser.parse_args()
    return args

def train(opt):
    torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    global_model = ActorCritic(num_states, num_actions)
    global_model.share_memory()
    if opt.load_from_previous_stage:
        if opt.stage == 1:
            previous_world = opt.world - 1
            previous_stage = 4
        else:
            previous_world = opt.world
            previous_stage = opt.stage - 1
        file_ = "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, previous_world, previous_stage)
        if os.path.isfile(file_):
            global_model.load_state_dict(torch.load(file_, map_location=torch.device('cpu')))
    optimizer = optim(global_model.parameters(), lr=opt.lr)
    processes = []
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer))
        process.start()
        processes.append(process)
    process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
    process.start()
    processes.append(process)
    for process in processes:
        process.join()

if __name__ == "__main__":
    opt = get_args()
    train(opt)
