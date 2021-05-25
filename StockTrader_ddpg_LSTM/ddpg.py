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
from visualizer import Visualizer
from actor import ActorNetwork
from critic import CriticNetwork
from replay_buffer import ReplayBuffer

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''


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
        self.next_sample = None
        self.training_data_idx = -1
        self.replay_memory = ReplayBuffer(300)
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        self.critic = value_network
        self.actor = policy_network
        self.tau = 0.01
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        self.memory_sample = []
        self.memory_next_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_target_policy = []
        self.memory_target_value = collections.deque()
        self.memory_target_action = []
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
        self.save_count = 0
        self.pre_pv = 0
    def init_policy_network(self, shared_network=None,
                            activation='sigmoid', loss='binary_crossentropy'):
        if self.rl_method == 'ddpg':
            self.actor = ActorNetwork(
                inp_dim=self.num_features,
                out_dim=self.agent.NUM_ACTIONS, lr=self.lr, tau=self.tau, num_steps=self.num_steps)
        if self.reuse_models and \
                os.path.exists(self.policy_network_path):
            self.actor.load_model(
                model_path=self.policy_network_path)

    def init_value_network(self, shared_network=None,
                           activation='linear', loss='mse'):
        if self.rl_method == 'ddpg':
            self.critic = CriticNetwork(
                inp_dim=self.num_features, lr=self.lr, tau=self.tau, num_steps=self.num_steps)
        if self.reuse_models and \
                os.path.exists(self.value_network_path):
            self.critic.load_model(
                model_path=self.value_network_path)

    def reset(self):
        self.sample = None
        self.next_sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_next_sample = []
        self.memory_action = []
        self.memory_target_policy = []
        self.memory_target_value = collections.deque()
        self.memory_target_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # Replay_buffer 초기회
        self.replay_memory.erase()
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_sample(self):
        self.environment.observe()
        making_sample = None
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            making_sample = self.training_data.iloc[
                self.training_data_idx].tolist()
            making_sample.extend(self.agent.get_states())
            return making_sample
        return None

    def build_next_sample(self):
        next_training_data_idx = self.training_data_idx  # copy idx
        making_sample = None

        if len(self.training_data) > next_training_data_idx + 1:
            next_training_data_idx += 1

            making_sample = self.training_data.iloc[
                self.training_data_idx].tolist()
            making_sample.extend(self.agent.get_states())
            return making_sample
        return None

    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    @abc.abstractmethod
    def train(self, batch_size, delayed_reward, discount_factor):
        pass

    def update_networks(self,
                        batch_size, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성
        loss = self.train(
            batch_size, delayed_reward, discount_factor)

        return loss

    def fit(self, batch_size, delayed_reward, discount_factor, full=False):
        batch_size = 10*self.num_steps
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
        self.memory_value = [np.array([np.nan] \
                                      * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                            + self.memory_value
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
            q_next_sample = collections.deque(maxlen=self.num_steps)
            q_action= collections.deque(maxlen=self.num_steps)
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
                sample = self.build_sample()
                if sample is None:
                    break
                next_sample = None
                if learning :
                    next_sample = self.build_next_sample()
                    if next_sample is None:
                        break

                # num_steps만큼 샘플 저장
                q_sample.append(sample)
                q_next_sample.append(next_sample)
                pred_policy1 = self.actor.predict2(sample)
                q_action.append(pred_policy1)
                if len(q_sample) < self.num_steps:
                    continue

                if len(q_next_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.actor is not None:
                    pred_policy = self.actor.predict(list(q_sample))
                if self.critic is not None:
                    pred_value = self.critic.predict(q_sample, q_action)

                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = \
                    self.agent.decide_action(pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = \
                    self.agent.act(action, confidence)
                if learning and immediate_reward > 0.01 or immediate_reward < -0.01:
                    self.replay_memory.add(q_sample, action, immediate_reward, q_next_sample)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(sample)
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                # self.memory_target_action.append(target_action)
                if self.actor is not None:
                    self.memory_value.append(pred_value)
                if self.critic is not None:
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
                if learning and self.replay_memory.count() > 10*self.num_steps:
                    self.fit(self.batch_size, delayed_reward, discount_factor)

            # 에포크 종료 후 학습
            # if learning:
            #    self.fit(
            #       self.agent.profitloss, discount_factor, full=True)
            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt >= 0:
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

            if epoch > num_epoches / 2 and self.agent.portfolio_value > self.pre_pv and self.agent.portfolio_value > self.agent.initial_balance:
                if self.critic is not None and \
                        self.value_network_path is not None:
                    self.critic.save_model(self.value_network_path)
                if self.actor is not None and \
                        self.policy_network_path is not None:
                    self.actor.save_model(self.policy_network_path)
                self.save_count += 1
                self.pre_pv = self.agent.portfolio_value

            if epoch % 100 == 0 and self.save_count == 0:
                if self.critic is not None and \
                        self.value_network_path is not None:
                    self.critic.save_model(self.value_network_path)
                if self.actor is not None and \
                        self.policy_network_path is not None:
                    self.actor.save_model(self.policy_network_path)

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
        if self.save_count == 0:
            if self.critic is not None and \
                    self.value_network_path is not None:
                self.critic.save_model(self.value_network_path)
            if self.actor is not None and \
                    self.policy_network_path is not None:
                self.actor.save_model(self.policy_network_path)


REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99


class DDPG(ReinforcementLearner):
    """docstring for DDPG"""

    def __init__(self, *args, shared_network=None,
                 value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'DDPG'  # name for uploading results
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=shared_network)

    def train(self, batch_size, delayed_reward, discount_factor):
        memory = self.replay_memory.get_batch(batch_size)

        samples = []
        actions = []
        rewards = []
        next_samples = []

        # (samples, actions, rewards, next_samples) = memory

        for i, (sample, action, reward, next_sample) in enumerate(memory):
            samples.append(sample)
            actions.append(action)
            rewards.append(reward)
            next_samples.append(next_sample)

        # Predict target q-values using target networks
        q_values = self.critic.target_predict(next_samples, self.actor.target_predict(next_samples))

        # Compute critic target
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            critic_target[i] = rewards[i] + delayed_reward + discount_factor * q_values[i]
        loss = 0
        # Train critic
        loss += self.critic.train_on_batch(samples, actions, critic_target)
        # Q-Value Gradients under Current Policy
        # actions = self.actor.predict(states) ##
        #

        #(actions)
        grads = self.critic.gradients(samples, actions)
        # print(np.array(grads).shape)
        #  print(np.array(grads).reshape((-1, 1, self.agent.NUM_ACTIONS)).shape)
        # Train actor
        self.actor.train(samples, np.array(grads).reshape((-1, self.num_steps, self.agent.NUM_ACTIONS)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

        return loss


