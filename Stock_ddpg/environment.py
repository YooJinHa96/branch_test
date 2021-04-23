import itertools
import numpy as np
import utils

STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

# 매매 수수료 및 세금
TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
# TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
# TRADING_CHARGE = 0  # 거래 수수료 미적용
TRADING_TAX = 0.0025  # 거래세 0.25%
# TRADING_TAX = 0  # 거래세 미적용

# 행동
ACTION_BUY = 0  # 매수
ACTION_SELL = 1  # 매도
ACTION_HOLD = 2  # 홀딩
# 인공 신경망에서 확률을 구할 행동들
ACTIONS = [ACTION_BUY, ACTION_SELL]
NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수


class Environment:
    PRICE_IDX = 4  # 종가의 위치

    def __init__(self, chart_data=None,balance=10000000,min_trading_unit=1, max_trading_unit=2,
        delayed_reward_threshold=.05):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1
        self.n_step, self.n_stock=self.chart_data.shape;

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        # 지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        self.initial_investment = balance
        self.cur_step=None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        # action space
        self.action_space = np.arange(3 ** self.n_stock)

        # action_permutations
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

        # calculate size of state
        self.state_dim = self.n_stock * 2 + 1

        # seed and start
        self.reset()
    def reset(self):
        self.observation = None
        self.idx = -1
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment

        self.portfolio_value = self.initial_investment
        self.base_portfolio_value = self.initial_investment
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

        return self._get_obs()

    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2 * self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def _step(self, action):
        assert action in self.action_space

        prev_val = self._get_val()

        # update price===go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]  # update price

        # perform the trade
        self._trade(action)

        # get the new value after
        cur_val = self._get_val()

        # reward is the Portfolio Value
        reward = cur_val - prev_val

        # done if we run out of data
        done = self.cur_step == self.n_step - 1

        # store the current portfolio value
        info = {'cur_val': cur_val}

        # confirm to the gym API
        return self._get_obs(), reward, done, info

    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand() / 2

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = (
                self.portfolio_value / self.base_portfolio_value
        )
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        if action == ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (
                    1 + self.TRADING_CHARGE) * self.min_trading_unit:
                # print("balance " + str(self.balance)+"right "+str(self.environment.get_price() * (
                # 1 + self.TRADING_CHARGE) * self.min_trading_unit))
                return False
        elif action == ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit -
                              self.min_trading_unit)),
            self.max_trading_unit - self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                    self.balance - curr_price * (1 + self.TRADING_CHARGE) \
                    * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                                curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) \
                            * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                    1 - (self.TRADING_TAX + self.TRADING_CHARGE)) \
                            * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price \
                               * self.num_stocks
        self.profitloss = (
                (self.portfolio_value - self.initial_balance) \
                / self.initial_balance
        )

        # 즉시 보상 - 수익률
        self.immediate_reward = self.profitloss

        # 지연 보상 - 익절, 손절 기준
        delayed_reward = 0
        self.base_profitloss = (
                (self.portfolio_value - self.base_portfolio_value) \
                / self.base_portfolio_value
        )
        if self.base_profitloss > self.delayed_reward_threshold or \
                self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward
