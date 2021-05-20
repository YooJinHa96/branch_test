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
import data_manager
import settings

from utils import sigmoid
from environment import Environment
from agent import Agent
from visualizer import Visualizer


class ReinforcementTester:
    def __init__(
        self,
        rl_method="rl",
        stock_code=None,
        start_date=None,
        end_date=None,
        ver="v3",
        min_trading_unit=1,
        max_trading_unit=2,
        delayed_reward_threshold=0.05,
        value_network_path="",
        net="dnn",
        num_steps=1,
        lr=0.001,
        value_network=None,
        output_path="",
        reuse_models=True,
    ):
        for stock in stock_code:
            chart_data, training_data = data_manager.load_data(
                os.path.join(settings.BASE_DIR, "data/{}/{}.csv".format(ver, stock)),
                start_date,
                end_date,
                ver,
            )
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.training_data = training_data
        self.value_network_path = value_network_path
        min_trading_unit = max(int(100 / chart_data.iloc[-1]["close"]), 1)
        max_trading_unit = max(int(1000 / chart_data.iloc[-1]["close"]), 1)

        self.environment = Environment(chart_data)
        self.agent = Agent(
            self.environment,
            min_trading_unit=min_trading_unit,
            max_trading_unit=max_trading_unit,
            delayed_reward_threshold=delayed_reward_threshold,
        )

        self.NUM_ACTIONS = 3
        self.distributional = False
        self.rl_method = rl_method

        self.value_network = value_network
        self.target_network = value_network
        self.reuse_models = reuse_models

        self.visualizer = Visualizer()

        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []

        self.num_steps = num_steps
        self.loss = 0.0
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        self.alpha = 0.6  # When using PER ratio of Prioritized buffer
        self.target_update_cnt = 5  # The Num of updating Target <- Value Nets

        self.output_path = output_path
        self.value_network.load_model(model_path=self.value_network_path)

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

    def run(
        self,
        num_epoches=1,
        balance=10000,
        discount_factor=0.9,
        start_epsilon=0.5,
        learning=False,
    ):
        info = (
            "[{code}] RL:{rl} Net:{net} LR:{lr} "
            "DF:{discount_factor} TU:[{min_trading_unit},"
            "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
                code=self.stock_code,
                rl=self.rl_method,
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

                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, "0")
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch

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
