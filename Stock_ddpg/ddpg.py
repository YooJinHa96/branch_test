import argparse
import os
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optims
from tensorboardX import SummaryWriter
import os
import logging
import abc
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer
'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="Pendulum-v0")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make(args.env_name)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp' + script_name + args.env_name +'./'


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', stock_code=None,
                 chart_data=None, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05,
                 net='dnn', num_steps=1, lr=0.001,
                 value_network=None, policy_network=None,
                 output_path='', reuse_models=True):
        # 인자 확인
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0
        # 강화학습 기법 설정
        self.rl_method = rl_method
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        #replay buffer
        self.replay_buffer = Replay_buffer()
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        # 로그 등 출력 경로
        self.output_path = output_path

    def init_value_network(self, shared_network=None,
                           activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)

        if self.reuse_models and \
                os.path.exists(self.value_network_path):
            self.value_network.load_model(
                model_path=self.value_network_path)

    def init_policy_network(self, shared_network=None,
                            activation='sigmoid', loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and \
                os.path.exists(self.policy_network_path):
            self.policy_network.load_model(
                model_path=self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[
                self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    def update_networks(self,
                        batch_size, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch(
            batch_size, delayed_reward, discount_factor)
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            return loss
        return None

    def fit(self, delayed_reward, discount_factor, full=False):
        batch_size = len(self.memory_reward) if full \
            else self.batch_size
        # 배치 학습 데이터 생성 및 신경망 갱신
        if batch_size > 0:
            _loss = self.update_networks(
                batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0

    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] \
                             * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) \
                                 + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] \
                                          * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                                + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] \
                                           * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                                 + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] \
                         * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches,
            epsilon=epsilon, action_list=Agent.ACTIONS,
            actions=self.memory_action,
            num_stocks=self.memory_num_stocks,
            outvals_value=self.memory_value,
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx,
            learning_idxes=self.memory_learning_idx,
            initial_balance=self.agent.initial_balance,
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(
            self.epoch_summary_dir,
            'epoch_summary_{}.png'.format(epoch_str))
        )

    def run(
            self, num_epoches=100, balance=10000000,
            discount_factor=0.9, start_epsilon=0.5, learning=True):
        info = "[{code}] RL:{rl} Net:{net} LR:{lr} " \
               "DF:{discount_factor} TU:[{min_trading_unit}," \
               "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
            code=self.stock_code, rl=self.rl_method, net=self.net,
            lr=self.lr, discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        )
        with self.lock:
            logging.info(info)

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(
            self.output_path, 'epoch_summary_{}'.format(
                self.stock_code))
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)

            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon \
                          * (1. - float(epoch) / (num_epoches - 1))
                self.agent.reset_exploration()
            else:
                epsilon = start_epsilon

            while True:
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(
                        list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(
                        list(q_sample))

                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = \
                    self.agent.decide_action(
                        pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = \
                    self.agent.act(action, confidence)

                next_state, reward, done, info = env.step(action)

                self.replay_buffer.push((list(q_sample), action,pred_value,pred_policy,reward))


                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

                # 지연 보상 발생된 경우 미니 배치 학습
                if learning and (delayed_reward != 0):
                    self.fit(delayed_reward, discount_factor)

            # 에포크 종료 후 학습
            if learning:
                self.fit(
                    self.agent.profitloss, discount_factor, full=True)

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt
            logging.info("[{}][Epoch {}/{}] Epsilon:{:.4f} "
                         "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                         "#Stocks:{} PV:{:,.0f} "
                         "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                self.stock_code, epoch_str, num_epoches, epsilon,
                self.exploration_cnt, self.itr_cnt,
                self.agent.num_buy, self.agent.num_sell,
                self.agent.num_hold, self.agent.num_stocks,
                self.agent.portfolio_value, self.learning_cnt,
                self.loss, elapsed_time_epoch))

            # 에포크 관련 정보 가시화
            self.visualize(epoch_str, num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logging.info("[{code}] Elapsed Time:{elapsed_time:.4f} "
                         "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                code=self.stock_code, elapsed_time=elapsed_time,
                max_pv=max_portfolio_value, cnt_win=epoch_win_cnt))

    def save_models(self):
        if self.value_network is not None and \
                self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and \
                self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (sampe,action,vaule,policy reward)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s, a, v, p, r = [], [], [], [], []

        for i in ind:
            S, A, V, P, R = self.storage[i]
            s.appnd(np.array(S, copy=False))
            a.append(np.array(A, copy=False))
            v.append(np.array(V, copy=False))
            p.append(np.array(P, copy=False))
            r.append(np.array(R, copy=False))

        return np.array(s), np.array(a), np.array(v), np.array(p), np.array(r).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(ReinforcementLearner):
    def __init__(self, *args, shared_network=None,
        value_network_path=None, policy_network_path=None,  state_dim, action_dim, max_action,**kwargs):
        super().__init__(*args, **kwargs)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optims.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optims.Adam(self.critic.parameters(), lr=1e-3)

        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

