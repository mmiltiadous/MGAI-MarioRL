import pickle
import random
from collections import deque
import matplotlib.pyplot as plt
import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *


def arrange(s):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)


class ReplayMemory(object):
    def __init__(self, N):
        self.memory = deque(maxlen=N)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(Model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 4, 2)
        self.layer3 = nn.Conv2d(64, 64, 3, 1)
        self.feature_map_size = self._get_feature_map_size(n_frame)
        self.fc = nn.Linear(self.feature_map_size, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)
        self.device = device
        self.apply(init_weights)

    def _get_feature_map_size(self, n_frame):
        with torch.no_grad():
            x = torch.zeros(1, n_frame, 84, 84)
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer3(x))
            return x.numel()

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.max(-1, True)[0])

        return q


def train(q, q_target, memory, batch_size, gamma, optimizer, device, update_target):
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    s = np.array(s).squeeze()
    s_prime = np.array(s_prime).squeeze()
    a_max = q(s_prime).argmax(1).unsqueeze(-1)
    r = torch.FloatTensor(r).unsqueeze(-1).to(device)
    done = torch.FloatTensor(done).unsqueeze(-1).to(device)
    if update_target:
        with torch.no_grad():
            y = r + gamma * q_target(s_prime).gather(1, a_max) * done
    else:
        with torch.no_grad():
            y = r + gamma * q(s_prime).gather(1, a_max) * done
    a = torch.tensor(a).unsqueeze(-1).to(device)
    q_value = torch.gather(q(s), dim=1, index=a.view(-1, 1).long())
    loss = F.smooth_l1_loss(q_value, y).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def copy_weights(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)


def main(env, q, q_target, optimizer, device, update_target=True):
    t = 0
    gamma = 0.99
    batch_size = 256
    N = 50000
    eps = 0.001
    memory = ReplayMemory(N)
    update_interval = 50
    print_interval = 10
    score_lst = []
    saved_episode = []
    total_score = 0.0
    loss = 0.0

    for k in range(1000):
        obs = env.reset()
        s = arrange(obs)
        done = False

        while not done:
            if eps > np.random.rand():
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    if device == "cpu":
                        a = q(s).argmax().item()
                    else:
                        a = q(s).cpu().argmax().item()
            s_prime, r, done, _ = env.step(a)
            s_prime = arrange(s_prime)
            total_score += r
            r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
            memory.push((s, float(r), int(a), s_prime, int(1 - done)))
            s = s_prime
            stage = env.unwrapped._stage
            if len(memory) > 2000:
                loss += train(
                    q,
                    q_target,
                    memory,
                    batch_size,
                    gamma,
                    optimizer,
                    device,
                    update_target,
                )
                t += 1
            if t % update_interval == 0 and update_target:
                copy_weights(q, q_target)
                torch.save(q.state_dict(), "mario_q.pth")
                torch.save(q_target.state_dict(), "mario_q_target.pth")

        if k % print_interval == 0:
            print(
                "%s |Epoch : %d | score : %f | loss : %.2f | stage : %d"
                % (
                    device,
                    k,
                    total_score / print_interval,
                    loss / print_interval,
                    stage,
                )
            )
            score_lst.append(total_score / print_interval)
            saved_episode.append(k)
            total_score = 0
            loss = 0.0

    plt.plot(saved_episode, score_lst)
    plt.ylabel("Mean of Total reward of every 10 episodes")
    plt.xlabel("Timestep")
    plt.show()


if __name__ == "__main__":
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = Model(n_frame, env.action_space.n, device).to(device)
    q_target = Model(n_frame, env.action_space.n, device).to(device)
    optimizer = optim.Adam(q.parameters(), lr=0.0001)
    print(device)

    main(env, q, q_target, optimizer, device, update_target=True)
