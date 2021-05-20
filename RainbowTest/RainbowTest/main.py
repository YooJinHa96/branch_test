import os
import sys
import logging
import argparse
import json

import settings
import utils
import data_manager
from Test import ReinforcementTester


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_code", nargs="+")  # Code os Stock
    parser.add_argument(
        "--ver", choices=["v1", "v2", "v3"], default="v3"
    )  # Set Data Versions
    parser.add_argument(
        "--rl_method",
        choices=[
            "dqn",  # DQN
            "rdqn",  # DQN with replay-buffer
            "rfdqn",  # Fixed DQN (2015)
            "rddqn",  # Double DQN (2015)
            "perdqn",  # DQN with Prioritized replay buffer (2016)
            "f_dqn",
            "ddqn",
            "p_ddqn",
            "c51",  # Distributional DQN using C51 (2017)
            "rainbow",  # Rainbow DQN (2017)
            "monkey",
        ],  # Set RL Algorithm
    )
    parser.add_argument(
        "--net",
        choices=[
            "dnn",
            "lstm",
            "cnn",
            "ndcnn",  # Noisy Dueling CNN
            "nddnn",  # Noisy Dueling DNN
            "dlstm",  # Dueling LSTM
            "dndlstm",  # Noisy Dueling Distributional LSTM
            "monkey",
        ],
        default="dnn",  # Set Networks
    )
    parser.add_argument("--num_steps", type=int, default=1)  # for n-step
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--start_epsilon", type=float, default=0)
    parser.add_argument("--balance", type=int, default=10000)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--delayed_reward_threshold", type=float, default=0.03)
    parser.add_argument(
        "--backend",  # Set backend framework
        choices=["tensorflow", "plaidml"],
        default="plaidml",
    )
    parser.add_argument("--output_name", default=utils.get_time_str())
    parser.add_argument("--value_network_name")
    parser.add_argument("--reuse_models", action="store_true")
    parser.add_argument("--learning", action="store_true")
    parser.add_argument("--start_date", default="20150104")
    parser.add_argument("--end_date", default="20181230")
    args = parser.parse_args()

    # Keras Backend 설정
    if args.backend == "tensorflow":
        os.environ["KERAS_BACKEND"] = "tensorflow"
    elif args.backend == "plaidml":
        os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    output_path = os.path.join(
        settings.BASE_DIR,
        "output/{}_{}_{}".format(args.output_name, args.rl_method, args.net),
    )
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, "params.json"), "w") as f:
        f.write(json.dumps(vars(args)))

    file_handler = logging.FileHandler(
        filename=os.path.join(output_path, "{}.log".format(args.output_name)),
        encoding="utf-8",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        handlers=[file_handler, stream_handler],
        level=logging.DEBUG,
    )

    from agent import Agent
    from learners import (
        ReinforcementLearner,
        ReplayDQNLearner,
        ReplayFixedDQNLearner,
        ReplayDoubleDQNLearner,
        ReplayPERDQNLearner,
        DQNLearner,
        DQNFixedLearner,
        DoubleDQNLearner,
        PrioritizedDoubleDQNLearner,
        ReplayC51DQNLearner,
        RainbowDQNLearner,
    )

    value_network_path = ""
    if args.value_network_name is not None:
        value_network_path = os.path.join(
            settings.BASE_DIR, "models/{}.h5".format(args.value_network_name)
        )
    else:
        value_network_path = os.path.join(
            output_path,
            "{}_{}_value_{}.h5".format(args.rl_method, args.net, args.output_name),
        )

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    for stock_code in args.stock_code:
        chart_data, training_data = data_manager.load_data(
            os.path.join(
                settings.BASE_DIR, "data/{}/{}.csv".format(args.ver, stock_code)
            ),
            args.start_date,
            args.end_date,
            ver=args.ver,
        )

        min_trading_unit = max(int(100 / chart_data.iloc[-1]["close"]), 1)
        max_trading_unit = max(int(1000 / chart_data.iloc[-1]["close"]), 1)

        common_params = {
            "rl_method": args.rl_method,
            "delayed_reward_threshold": args.delayed_reward_threshold,
            "net": args.net,
            "num_steps": args.num_steps,
            "lr": args.lr,
            "output_path": output_path,
            "reuse_models": args.reuse_models,
        }

        learner = None
        common_params.update(
            {
                "stock_code": stock_code,
                "chart_data": chart_data,
                "training_data": training_data,
                "min_trading_unit": min_trading_unit,
                "max_trading_unit": max_trading_unit,
            }
        )
        if args.rl_method == "dqn":
            learner = DQNLearner(
                **{**common_params, "value_network_path": value_network_path}
            )
        elif args.rl_method == "rdqn":
            learner = ReplayDQNLearner(
                **{**common_params, "value_network_path": value_network_path}
            )
        elif args.rl_method == "rfdqn":
            learner = ReplayFixedDQNLearner(
                **{**common_params, "value_network_path": value_network_path}
            )
        elif args.rl_method == "rddqn":
            learner = ReplayDoubleDQNLearner(
                **{**common_params, "value_network_path": value_network_path}
            )
        elif args.rl_method == "perdqn":
            learner = ReplayPERDQNLearner(
                **{**common_params, "value_network_path": value_network_path}
            )
        elif args.rl_method == "f_dqn":
            learner = DQNFixedLearner(
                **{**common_params, "value_network_path": value_network_path}
            )
        elif args.rl_method == "ddqn":
            learner = DoubleDQNLearner(
                **{**common_params, "value_network_path": value_network_path}
            )
        elif args.rl_method == "p_ddqn":
            learner = PrioritizedDoubleDQNLearner(
                **{**common_params, "value_network_path": value_network_path}
            )
        elif args.rl_method == "c51":
            learner = ReplayC51DQNLearner(
                **{**common_params, "value_network_path": value_network_path}
            )
        elif args.rl_method == "rainbow":
            learner = RainbowDQNLearner(
                **{**common_params, "value_network_path": value_network_path}
            )
        elif args.rl_method == "monkey":
            args.net = args.rl_method
            args.num_epoches = 1
            args.discount_factor = None
            args.start_epsilon = 1
            args.learning = False
            learner = ReinforcementLearner(**common_params)

        if learner is not None:
            learner.run(
                balance=args.balance,
                num_epoches=args.num_epoches,
                discount_factor=args.discount_factor,
                start_epsilon=args.start_epsilon,
                learning=args.learning,
            )
            learner.save_models()
