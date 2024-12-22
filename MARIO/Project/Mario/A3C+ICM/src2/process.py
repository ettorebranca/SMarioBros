import torch
from .env import create_train_env
from .model import ActorCritic, ICM
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit
import logging
import numpy as np

def local_train(index, opt, global_model, optimizer, icm=None,  save=False):
    torch.manual_seed(123 + index)
    completions = 0
    start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path + str(index))
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCritic(num_states, num_actions)
    icm = ICM(num_inputs=4, num_actions=num_actions) 
    local_model.train()
    icm.train()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    curr_episode = 0
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=opt.lr)  ### icm parameters not shared among processes
    while True:
        if save and curr_episode % opt.save_interval == 0 and curr_episode > 0:
            torch.save(global_model.state_dict(), "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
            torch.save(icm.state_dict(), "{}/icm_{}_{}".format(opt.saved_path, opt.world, opt.stage))
        print("Process {}. Episode {}".format(index, curr_episode))
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        h_0 = torch.zeros((1, 512), dtype=torch.float) if done else h_0.detach()
        c_0 = torch.zeros((1, 512), dtype=torch.float) if done else c_0.detach()
        log_policies, values, rewards, entropies = [], [], [], []
        intrinsic_rewards = []
        for _ in range(opt.num_local_steps):
            curr_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=-1)
            log_policy = F.log_softmax(logits, dim=-1)
            entropy = -(policy * log_policy).sum(-1, keepdim=True)
            m = Categorical(policy)
            action = m.sample().item()
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(next_state)
            if info["flag_get"]:
                completions += 1
            env.render()
            if curr_step > opt.num_global_steps:
                done = True
            if done:
                curr_step = 0
                state = torch.from_numpy(env.reset())
            ###################################################
            # Calculation of intrinsic reward 
            icm_input_state = state.view(1, -1, 84, 84) 
            icm_input_next_state = next_state.view(1, -1, 84, 84)
            action_tensor = torch.tensor([action], dtype=torch.long)
            _, phi_next, pred_action, pred_phi_next = icm(icm_input_state, icm_input_next_state, action_tensor.unsqueeze(0))
            phi_next_flat = phi_next.view(phi_next.size(0), -1)  
            intrinsic_reward = F.mse_loss(pred_phi_next, phi_next_flat.detach(), reduction='none').mean()   
            # Total reward is a combination of intrinsic and extrinsic rewards  
            total_reward = reward + opt.eta * intrinsic_reward.item()   ### scaling factor for ir
            #print(f"Reward: {reward}")
            #print(f"Intrinsic Reward: {intrinsic_reward.item()}")
            ####################################################
            state = next_state if not done else torch.from_numpy(env.reset())
            log_policies.append(log_policy[0, action])
            values.append(value)
            rewards.append(total_reward)
            entropies.append(entropy)
            intrinsic_rewards.append(intrinsic_reward)
            if done:
                break
        R = torch.zeros((1, 1), dtype=torch.float)
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)
        actor_loss, critic_loss, entropy_loss = 0, 0, 0
        gae = torch.zeros(1, 1, dtype=torch.float)
        next_value = R
        tot_reward=0
        for value, log_policy, reward, entropy in zip(reversed(values), reversed(log_policies), reversed(rewards), reversed(entropies)):
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss - log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
            tot_reward+=reward
        total_loss = actor_loss + 0.5 * critic_loss + opt.beta * entropy_loss
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        writer.add_scalar("Train_{}/Reward".format(index), tot_reward, curr_episode)
        writer.add_scalar("Train_{}/Actor_Loss".format(index), actor_loss, curr_episode)
        writer.add_scalar("Train_{}/Critic_Loss".format(index), critic_loss, curr_episode)
        writer.add_scalar("Train_{}/Completions".format(index), completions, curr_episode)
        writer.flush()
        # Update global model
        optimizer.zero_grad()
        total_loss.backward()
        for param in local_model.parameters():
            param.requires_grad = True
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=100)
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad
        optimizer.step()
        # Update ICM
        icm_optimizer.zero_grad()
        inverse_loss = F.cross_entropy(pred_action, action_tensor)
        forward_loss = F.mse_loss(pred_phi_next, phi_next_flat.detach())
        icm_loss = inverse_loss + opt.lambda_forward * forward_loss
        icm_loss.backward()
        icm_optimizer.step()
        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            logging.error("Training process %s terminated", str(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
                logging.error('The code runs for %.2f s ', (end_time - start_time))
            return

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
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)