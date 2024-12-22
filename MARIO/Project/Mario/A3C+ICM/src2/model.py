import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)     
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)      
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx): 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx)) 
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx

class ICM(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ICM, self).__init__()
        self.num_actions = num_actions
        self.feature = nn.Sequential(   # Feature extractor
            nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),          
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),          
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),         
            nn.ReLU())
        self.flattened_size = 32 * 6 * 6  
        self.inverse_fc = nn.Sequential(              # Inverse model
            nn.Linear(self.flattened_size * 2, 256),  # 2 per la concatenazione di phi_state e phi_next
            nn.ReLU(),
            nn.Linear(256, num_actions))
        self.forward_fc = nn.Sequential(                        # Forward model
            nn.Linear(self.flattened_size + num_actions, 256),  # Aggiungere num_actions
            nn.ReLU(),
            nn.Linear(256, self.flattened_size))

    def forward(self, state, next_state, action):
        # Feature encoding
        phi_state = self.feature(state)  
        phi_next = self.feature(next_state) 
        phi_state_flat = phi_state.view(phi_state.size(0), -1)  # Flatten
        phi_next_flat = phi_next.view(phi_next.size(0), -1) 
        # Inverse model
        phi_concat = torch.cat((phi_state_flat, phi_next_flat), dim=1)  
        pred_action = self.inverse_fc(phi_concat)  
        # Forward model
        action = action.squeeze(1) 
        action_one_hot = F.one_hot(action, num_classes= self.num_actions).float()  
        forward_input = torch.cat((phi_state_flat, action_one_hot), dim=1)  
        forward_input_flat = forward_input.view(forward_input.size(0), -1)
        pred_phi_next = self.forward_fc(forward_input_flat)  
        return phi_state, phi_next, pred_action, pred_phi_next