import torch
from .env import create_train_env
from .model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit
import logging
import numpy as np

def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    completions = 0
    writer = SummaryWriter(opt.log_path + str(index))
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCritic(num_states, num_actions)
    local_model.train()
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    curr_step = 0
    curr_episode = 0
    while True:
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(), "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
            print("Process {}. Episode {}".format(index, curr_episode))
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        trajectory = []
        episode_reward = 0
        for _ in range(opt.num_local_steps):
            curr_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=-1)
            m = Categorical(policy)
            action = m.sample().item()
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            if info["flag_get"]:
                completions += 1
            env.render() 
            trajectory.append((state, action, reward, value, done, info))
            state = torch.from_numpy(next_state)
            if curr_step > opt.num_global_steps:
                done = True
            if done:
                curr_step = 0
                state = torch.from_numpy(env.reset())
            if done:
                break
        # Reformulate trajectory using diverse goals and prioritize high-quality ones
        hindsight_trajectories = reformulate_trajectory_multiple(trajectory, num_goals=3)
        hindsight_trajectories = prioritize_trajectories(hindsight_trajectories, reward_threshold=200)
        # Compute loss
        total_loss = 0
        for traj in hindsight_trajectories:
            total_loss += compute_loss(traj, local_model, opt)
        total_loss = total_loss.sum() 
        writer.add_scalar("Train_{}/Loss".format(index), total_loss.item(), curr_episode)
        writer.add_scalar("Train_{}/Completions".format(index), completions, curr_episode)
        writer.add_scalar("Train_{}/Reward".format(index), episode_reward, curr_episode) 
        writer.flush()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 100)
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()

def reformulate_trajectory_multiple(trajectory, num_goals=3):
    """Reformulate trajectory with multiple goals: intermediate and final goals."""
    reformulated_trajectories = []
    for _ in range(num_goals):
        # Select a random step in the trajectory (starting from x% of the trajectory) as the goal
        t_goal = np.random.randint(len(trajectory)*0.75, len(trajectory))
        goal_x_pos = trajectory[t_goal][5].get("x_pos", 0)  # Use info from the randomly selected step
        hindsight_goal = {"x_pos": goal_x_pos}
        reformulated_trajectory = []
        for t, (state, action, reward, value, done, info) in enumerate(trajectory):
            hindsight_reward = calculate_reward_based_on_x_pos(info.get("x_pos", 0), hindsight_goal)
            combined_reward = reward + hindsight_reward
            reformulated_trajectory.append((state, action, combined_reward, value, done, hindsight_goal))
        reformulated_trajectories.append(reformulated_trajectory)
    return reformulated_trajectories

def calculate_reward_based_on_x_pos(current_x_pos, goal):
    goal_x_pos = goal["x_pos"]
    if current_x_pos > goal_x_pos:
        return 50.0  #positive reward if agent overtakes goal 
    else:
        # progressive penalty based on distance from goal
        distance = goal_x_pos - current_x_pos
        return -0.1 * distance
    
def prioritize_trajectories(trajectories, reward_threshold=200):
    """Prioritize high-quality trajectories based on total reward."""
    prioritized = [traj for traj in trajectories if sum([r for _, _, r, _, _, _ in traj]) > reward_threshold]
    return prioritized if prioritized else trajectories  # Return original if no high-quality found

def compute_loss(trajectory, model, opt):
    R = 0
    actor_loss, critic_loss, entropy_loss = 0, 0, 0
    h_0 = torch.zeros((1, 512), dtype=torch.float)  
    c_0 = torch.zeros((1, 512), dtype=torch.float)  
    gae = torch.zeros(1, 1, dtype=torch.float)  
    advantages = []
    for state, action, reward, value, _, _ in reversed(trajectory):
        R = reward + opt.gamma * R  
        td_error = reward + opt.gamma * value.item() - value.item() 
        gae = td_error + opt.gamma * opt.tau * gae
        advantages.append(gae)
    advantages = torch.flip(torch.stack(advantages), dims=[0])  
    # Compute loss for each step of the trajectory
    for t, (state, action, reward, value, _, _) in enumerate(trajectory):
        advantage = advantages[t]
        critic_loss = critic_loss + 0.5 * advantage ** 2
        logits, _, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=-1)
        log_policy = F.log_softmax(logits, dim=-1)
        actor_loss = actor_loss + (-log_policy[0, action]* advantage)
        entropy_loss = entropy_loss - torch.sum(policy * log_policy)
    total_loss = actor_loss + 0.5*critic_loss - opt.beta * entropy_loss
    return total_loss

def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCritic(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    total = 0
    while True:
        curr_step += 1
        local_model.load_state_dict(global_model.state_dict())
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()
        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        total = total + reward
        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            print("TEST reward: " + str(total))
            total = 0
            curr_step = 0
            local_model.load_state_dict(global_model.state_dict())
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)