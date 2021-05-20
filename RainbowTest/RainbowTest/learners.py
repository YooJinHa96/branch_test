import os
import logging
import abc
import collections
import threading
import time
import numpy as np
import heapq
import random
import math
import Replay_Memory

from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import (
    Network,
    DNN,
    NDDNN,
    CNN,
    NDCNN,
    LSTMNetwork,
    DuelingNoisyLSTM,
    DNDLSTM,
)
from visualizer import Visualizer


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(
        self,
        rl_method="rl",
        stock_code=None,
        chart_data=None,
        training_data=None,
        min_trading_unit=1,
        max_trading_unit=2,
        delayed_reward_threshold=0.05,
        net="dnn",
        num_steps=1,
        lr=0.001,
        value_network=None,
        output_path="",
        reuse_models=True,
    ):
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0

        self.rl_method = rl_method

        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        self.agent = Agent(
            self.environment,
            min_trading_unit=min_trading_unit,
            max_trading_unit=max_trading_unit,
            delayed_reward_threshold=delayed_reward_threshold,
        )
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]

        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.target_network = value_network
        self.reuse_models = reuse_models
        self.distributional = False
        self.is_replay = False
        self.NUM_ACTIONS = 3

        self.visualizer = Visualizer()

        self.replay_memory = Replay_Memory.Replay_Memory(1000)
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_target = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        self.per_buffer = []
        self.replay_cnt = 0
        self.delay_cnt = 0
        self.loss = 0.0
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        self.alpha = 0.6  # When using PER ratio of Prioritized buffer
        self.target_update_cnt = 5  # The Num of updating Target <- Value Nets

        self.output_path = output_path

    def init_value_network(
        self,
        activation="linear",
        loss="mse",
    ):
        if self.net == "dnn":
            self.value_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr,
                activation=activation,
                loss=loss,
            )
        elif self.net == "lstm":
            self.value_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr,
                num_steps=self.num_steps,
                activation=activation,
                loss=loss,
            )
        elif self.net == "dlstm":
            self.value_network = DuelingNoisyLSTM(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr,
                num_steps=self.num_steps,
                activation=activation,
                loss=loss,
            )
        elif self.net == "cnn":
            self.value_network = CNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr,
                num_steps=self.num_steps,
                activation=activation,
                loss=loss,
            )
        elif self.net == "ndcnn":
            self.value_network = NDCNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr,
                num_steps=self.num_steps,
                activation=activation,
                loss=loss,
            )
        elif self.net == "nddnn":
            self.value_network = NDDNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr,
                activation=activation,
                loss=loss,
            )
        elif self.net == "dndlstm":
            self.value_network = DNDLSTM(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr,
                num_steps=self.num_steps,
                activation=activation,
                loss=loss,
            )
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        self.environment.reset()
        self.agent.reset()
        self.visualizer.clear([0, len(self.chart_data)])
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        self.memory_target = []
        self.per_buffer = []
        self.loss = 0.0
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

        self.replay_cnt = 0
        self.delay_cnt = 0

        self.num_atoms = 51  # for categorical DQN -> C51 algorithm
        self.v_max = 30
        self.v_min = -10
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    def build_next_sample(self):
        next_training_data_idx = self.training_data_idx
        if len(self.training_data) > next_training_data_idx + 1:
            next_training_data_idx += 1
            next_sample = self.training_data.iloc[next_training_data_idx].tolist()
            next_sample.extend(self.agent.get_states())
            return next_sample
        return None

    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    def update_networks(
        self,
        batch_size,
        delayed_reward,
        discount_factor,
    ):
        if self.is_replay is not False:
            x, y_value = self.get_batch(
                self.replay_memory.random_data(batch_size),
                batch_size,
                discount_factor,
            )
        else:
            x, y_value = self.get_batch(batch_size, delayed_reward, discount_factor)
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                loss += self.value_network.train_on_batch(x, y_value)
            return loss
        return None

    def fit(self, delayed_reward, discount_factor, full=False):
        batch_size = len(self.memory_reward) if full else self.batch_size
        if batch_size > 0:
            _loss = self.update_networks(batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0

    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] * (
            self.num_steps - 1
        ) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] * (
                self.num_steps - 1
            ) + self.memory_value
        self.memory_pv = [self.agent.initial_balance] * (
            self.num_steps - 1
        ) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str,
            num_epoches=num_epoches,
            epsilon=epsilon,
            action_list=Agent.ACTIONS,
            actions=self.memory_action,
            num_stocks=self.memory_num_stocks,
            outvals_value=self.memory_value,
            exps=self.memory_exp_idx,
            learning_idxes=self.memory_learning_idx,
            initial_balance=self.agent.initial_balance,
            pvs=self.memory_pv,
            distributional=self.distributional,
        )
        self.visualizer.save(
            os.path.join(
                self.epoch_summary_dir, "epoch_summary_{}.png".format(epoch_str)
            )
        )

    def run(
        self,
        num_epoches=100,
        balance=10000,
        discount_factor=0.9,
        start_epsilon=0.5,
        learning=True,
    ):
        info = (
            "[{code}] RL:{rl} Net:{net} LR:{lr} "
            "DF:{discount_factor} TU:[{min_trading_unit},"
            "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
                code=self.stock_code,
                rl=self.rl_method,
                net=self.net,
                lr=self.lr,
                discount_factor=discount_factor,
                min_trading_unit=self.agent.min_trading_unit,
                max_trading_unit=self.agent.max_trading_unit,
                delayed_reward_threshold=self.agent.delayed_reward_threshold,
            )
        )
        with self.lock:
            logging.info(info)
        time_start = time.time()

        self.visualizer.prepare(self.environment.chart_data, info)

        self.epoch_summary_dir = os.path.join(
            self.output_path, "epoch_summary_{}".format(self.stock_code)
        )
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        self.agent.set_balance(balance)

        max_portfolio_value = 0
        epoch_win_cnt = 0
        epoch_num = 0
        for epoch in range(num_epoches):
            time_start_epoch = time.time()
            epoch_num += 1
            q_sample = collections.deque(maxlen=self.num_steps)
            n_sample = collections.deque(maxlen=self.num_steps)
            self.reset()

            if learning:
                epsilon = start_epsilon * (1.0 - float(epoch) / (num_epoches - 1))
                self.agent.reset_exploration()
            else:
                epsilon = start_epsilon
                self.agent.reset_exploration(alpha=0)

            lr_cnt = 0
            while True:
                cur_sample = self.build_sample()
                if cur_sample is None:
                    break

                q_sample.append(cur_sample)
                next_sample = self.build_next_sample()
                n_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                pred_value = None
                target_value = None

                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.target_network is not None:
                    target_value = self.target_network.predict(list(q_sample))

                action, confidence, exploration = self.agent.decide_action(
                    pred_value, epsilon, self.distributional
                )

                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                self.replay_memory.push(
                    list(q_sample), action, immediate_reward, list(n_sample)
                )
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)

                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.target_network is not None:
                    self.memory_target.append(target_value)

                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

                if self.is_replay is not False:
                    if learning and self.replay_memory.get_len() > 200:
                        self.batch_size = 3
                        self.fit(immediate_reward, discount_factor)
                        lr_cnt += 1
                        if lr_cnt % self.target_update_cnt == 0:
                            self.target_network = self.value_network
                else:
                    if learning and (delayed_reward != 0):
                        self.fit(delayed_reward, discount_factor)
                        lr_cnt += 1
                        if lr_cnt % self.target_update_cnt == 0:
                            self.target_network = self.value_network
            if learning:
                self.fit(self.agent.profitloss, discount_factor, full=True)

            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, "0")
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt

            logging.info(
                "[{}][Epoch {}/{}] Epsilon:{:.4f} "
                "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                "#Stocks:{} PV:{:,.0f} "
                "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                    self.stock_code,
                    epoch_str,
                    num_epoches,
                    epsilon,
                    self.exploration_cnt,
                    self.itr_cnt,
                    self.agent.num_buy,
                    self.agent.num_sell,
                    self.agent.num_hold,
                    self.agent.num_stocks,
                    self.agent.portfolio_value,
                    self.learning_cnt,
                    self.loss,
                    elapsed_time_epoch,
                )
            )

            self.visualize(epoch_str, num_epoches, epsilon)

            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        time_end = time.time()
        elapsed_time = time_end - time_start

        with self.lock:
            logging.info(
                "[{code}] Elapsed Time:{elapsed_time:.4f} "
                "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                    code=self.stock_code,
                    elapsed_time=elapsed_time,
                    max_pv=max_portfolio_value,
                    cnt_win=epoch_win_cnt,
                )
            )

    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)


class ReplayDQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()
        self.is_replay = True

    def get_batch(self, replay_memory, batch_size, discount_factor):
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        cnt = 0

        for i, (sample, action, reward, next_sample) in enumerate(replay_memory):
            x[i] = sample
            if next_sample[self.num_steps - 1] is not None:
                y_value[i - cnt, action] = reward + discount_factor * np.amax(
                    self.value_network.predict(next_sample)
                )
            else:
                cnt += 1
        return x, y_value


class ReplayFixedDQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()
        self.is_replay = True
        self.target_network = self.value_network

    def get_batch(self, replay_memory, batch_size, discount_factor):
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        cnt = 0

        for i, (sample, action, reward, next_sample) in enumerate(replay_memory):
            x[i] = sample
            if next_sample[self.num_steps - 1] is not None:
                y_value[i - cnt, action] = reward + discount_factor * np.amax(
                    self.target_network.predict(next_sample)
                )
            else:
                cnt += 1
        return x, y_value


class ReplayDoubleDQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()
        self.is_replay = True
        self.target_network = self.value_network

    def get_batch(self, replay_memory, batch_size, discount_factor):
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        cnt = 0

        for i, (sample, action, reward, next_sample) in enumerate(replay_memory):
            x[i] = sample
            if next_sample[self.num_steps - 1] is not None:
                y_value[i - cnt, action] = (
                    reward
                    + discount_factor * self.target_network.predict(next_sample)[action]
                )
            else:
                cnt += 1
        return x, y_value


class ReplayPERDQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()
        self.is_replay = True
        self.target_network = self.value_network

    def get_batch(self, replay_memory, batch_size, discount_factor):
        for i, (state, action, reward, next_state) in enumerate(replay_memory):
            if next_state[self.num_steps - 1] is not None:
                target = (
                    reward
                    + discount_factor
                    * self.target_network.predict(next_state)[
                        np.argmax(self.value_network.predict(next_state))
                    ]
                )
            else:
                target = reward
            q_values = self.value_network.predict(state)[action]
            td_error = q_values - target

            t = (state, action, reward, next_state)
            heapq.heappush(self.per_buffer, (-td_error, t))

        heapq.heapify(self.per_buffer)

        prioritization = int(batch_size * self.alpha)
        batch_prioritized = heapq.nsmallest(prioritization, self.per_buffer)
        batch_uniform = random.sample(self.per_buffer, batch_size - prioritization)
        per_batch = batch_prioritized + batch_uniform
        per_batch = [e for (_, e) in per_batch]

        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        cnt = 0
        for i, (sample, action, reward, next_state) in enumerate(per_batch):
            x[i] = sample
            if next_state[self.num_steps - 1] is not None:
                y_value[i - cnt, action] = (
                    reward
                    + discount_factor
                    * self.target_network.predict(next_state)[
                        np.argmax(self.value_network.predict(next_state))
                    ]
                )
            else:
                cnt += 1
        return x, y_value


class ReplayC51DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()
        self.is_replay = True
        self.distributional = True
        self.target_network = self.value_network

    def get_batch(self, replay_memory, batch_size, discount_factor):
        m_prob = [
            np.zeros((batch_size, self.num_atoms))
            for i in range(self.agent.NUM_ACTIONS)
        ]
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        cnt = 0

        for i, (sample, action, reward, next_state) in enumerate(replay_memory):
            x[i] = sample
            if next_state[self.num_steps - 1] is not None:
                z = self.value_network.predict(next_state)
                z_ = self.target_network.predict(next_state)
                z_concat = np.vstack(z)
                q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
                next_actions = np.argmax(q)
                for j in range(self.num_atoms):
                    Tz = min(
                        self.v_max, max(self.v_min, reward + discount_factor * reward)
                    )
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action][i][int(m_l)] += z_[next_actions][0][j] * (m_u - bj)
                    m_prob[action][i][int(m_u)] += z_[next_actions][0][j] * (bj - m_l)
            else:
                z = self.value_network.predict(sample)
                z_ = self.target_network.predict(sample)
                z_concat = np.vstack(z)
                q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
                next_actions = np.argmax(q)
                Tz = min(self.v_max, max(self.v_min, reward))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action][i][int(m_l)] += z_[next_actions][0][
                    self.num_atoms - 1
                ] * (m_u - bj)
                m_prob[action][i][int(m_u)] += z_[next_actions][0][
                    self.num_atoms - 1
                ] * (bj - m_l)
        return x, m_prob


class RainbowDQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()
        self.is_replay = True
        self.distributional = True
        self.target_network = self.value_network

    def get_batch(self, replay_memory, batch_size, discount_factor):
        for i, (state, action, reward, next_state) in enumerate(replay_memory):
            if next_state[self.num_steps - 1] is not None:
                target = self.target_network.predict(next_state)
            else:
                target = self.target_network.predict(state)
            q_values = self.value_network.predict(state)
            td_error = np.max(q_values) - np.max(target)

            t = (state, action, reward, next_state)
            heapq.heappush(self.per_buffer, (-td_error, t))

        heapq.heapify(self.per_buffer)

        prioritization = int(batch_size * self.alpha)
        batch_prioritized = heapq.nsmallest(prioritization, self.per_buffer)
        batch_uniform = random.sample(self.per_buffer, batch_size - prioritization)
        per_batch = batch_prioritized + batch_uniform
        per_batch = [e for (_, e) in per_batch]
        m_prob = [
            np.zeros((batch_size, self.num_atoms))
            for i in range(self.agent.NUM_ACTIONS)
        ]
        x = np.zeros((batch_size, self.num_steps, self.num_features))

        for i, (sample, action, reward, next_state) in enumerate(per_batch):
            x[i] = sample
            if next_state[self.num_steps - 1] is not None:
                z = self.value_network.predict(next_state)
                z_ = self.target_network.predict(next_state)
                z_concat = np.vstack(z)
                q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
                next_actions = np.argmax(q)
                for j in range(self.num_atoms):
                    Tz = min(
                        self.v_max, max(self.v_min, reward + discount_factor * reward)
                    )
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action][i][int(m_l)] += z_[next_actions][0][j] * (m_u - bj)
                    m_prob[action][i][int(m_u)] += z_[next_actions][0][j] * (bj - m_l)
            else:
                z = self.value_network.predict(sample)
                z_ = self.target_network.predict(sample)
                z_concat = np.vstack(z)
                q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
                next_actions = np.argmax(q)
                Tz = min(self.v_max, max(self.v_min, reward))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action][i][int(m_l)] += z_[next_actions][0][
                    self.num_atoms - 1
                ] * (m_u - bj)
                m_prob[action][i][int(m_u)] += z_[next_actions][0][
                    self.num_atoms - 1
                ] * (bj - m_l)
        return x, m_prob


class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            value_max_next = value.max()
            reward_next = reward
        return x, y_value


class DQNFixedLearner(ReinforcementLearner):  # Using 2015 DQN
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()
        self.target_network = self.value_network

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_target[-batch_size:]),  # target_network
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, target, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = target
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            value_max_next = target.max()
            reward_next = reward
        return x, y_value


class DoubleDQNLearner(ReinforcementLearner):  # Using Double DQN
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()
        self.target_network = self.value_network

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_target[-batch_size:]),  # target_network
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, target, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = target
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            value_max_next = target[action]
            reward_next = reward
        return x, y_value


class PrioritizedDoubleDQNLearner(
    ReinforcementLearner
):  # Stochastic Prioritzed Experience Replay by Using Double DQN
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()
        self.target_network = self.value_network

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_target[-batch_size:]),  # target_network
            reversed(self.memory_reward[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
        )

        value_max_next_for_p = 0
        reward_next_for_p = self.memory_reward[-1]
        for i, (sample, action, target, reward, value) in enumerate(
            memory
        ):  # calculate td_error & reinitialize buffer
            r2 = delayed_reward + reward_next_for_p - reward * 2
            target_for_p = r2 + discount_factor * value_max_next_for_p
            value_max_next_for_p = target[action]
            reward_next_next_for_p = reward
            td_error = value.max() - target_for_p
            t = (sample, action, target, reward)
            heapq.heappush(self.per_buffer, (-td_error, t))
        self.per_buffer = self.per_buffer[:-1]
        heapq.heapify(self.per_buffer)

        prioritization = int(batch_size * self.alpha)
        batch_prioritized = heapq.nsmallest(prioritization, self.per_buffer)
        batch_uniform = random.sample(self.per_buffer, batch_size - prioritization)
        per_batch = batch_prioritized + batch_uniform
        per_batch = [e for (_, e) in per_batch]

        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, target, reward) in enumerate(per_batch):
            x[i] = sample
            y_value[i] = target
            r = delayed_reward + reward_next - reward * 2
            y_value[i, action] = r + discount_factor * value_max_next
            value_max_next = target[action]
            reward_next = reward
        return x, y_value
