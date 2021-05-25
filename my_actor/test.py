import os
import sys
import logging
import argparse
import json
import settings
import utils
import data_manager
from learners import ReinforcementLearner

class Test:
    def __init__(self, stock_code, rl_method, start_date, end_date):
        parser = argparse.ArgumentParser()
        parser.add_argument('--stock_code', nargs='+', default='xon')
        parser.add_argument('--ver', choices=['v1', 'v2','v3'], default='v3')
        parser.add_argument('--rl_method',
            choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'ddpg','td3'])
        parser.add_argument('--net',
            choices=['dnn', 'lstm', 'cnn','actorcritic'], default='actorcritic')
        parser.add_argument('--num_steps', type=int, default=1)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--discount_factor', type=float, default=0.9)
        parser.add_argument('--start_epsilon', type=float, default=0)
        parser.add_argument('--balance', type=int, default=10000)
        parser.add_argument('--num_epoches', type=int, default=1)
        parser.add_argument('--delayed_reward_threshold',
        type=float, default=0.05)
        parser.add_argument('--backend',
            choices=['tensorflow', 'plaidml'], default='tensorflow')
        parser.add_argument('--output_name', default=utils.get_time_str())
        parser.add_argument('--policy_network_name')
        parser.add_argument('--reuse_models', action='store_true', default=True)
        parser.add_argument('--learning', action='store_true')
        parser.add_argument('--start_date', default='20200101')
        parser.add_argument('--end_date', default='20201230')
        args = parser.parse_args()

        # Keras Backend 설정
        if args.backend == 'tensorflow':
          os.environ['KERAS_BACKEND'] = 'tensorflow'
        elif args.backend == 'plaidml':
            os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

       # 출력 경로 설정
        output_path = os.path.join(settings.BASE_DIR,
            'output/{}_{}_{}'.format(args.output_name, rl_method , args.net))
        if not os.path.isdir(output_path):
            os.makedirs(output_path)


    # 파라미터 기록
        with open(os.path.join(output_path, 'params.json'), 'w') as f:
            f.write(json.dumps(vars(args)))

        # 로그 기록 설정
        file_handler = logging.FileHandler(filename=os.path.join(
            output_path, "{}.log".format(args.output_name)), encoding='utf-8')
        stream_handler = logging.StreamHandler(sys.stdout)
        file_handler.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.INFO)
        logging.basicConfig(format="%(message)s",
            handlers=[file_handler, stream_handler], level=logging.DEBUG)


         # 모델 경로 준비
        value_network_path = ''
        policy_network_path = ''
    #if args.value_network_name is not None:
     #   value_network_path = os.path.join(settings.BASE_DIR,
      #      'models/{}.h5'.format(args.value_network_name))
   # else:
    #    value_network_path = os.path.join(
     #       output_path, '{}_{}_value_{}.h5'.format(
      #          args.rl_method, args.net, args.output_name))
        if args.policy_network_name is not None:
            policy_network_path = os.path.join(settings.BASE_DIR,
            'models/{}.h5'.format(args.policy_network_name))
        else:
            policy_network_path = os.path.join(
                output_path, '{}_{}_policy_{}.h5'.format(
                rl_method, args.net, args.output_name))



            # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
                os.path.join(settings.BASE_DIR,
                'data/{}/{}.csv'.format(args.ver, stock_code)),
                args.start_date, args.end_date, ver=args.ver)

        # 최소/최대 투자 단위 설정
        min_trading_unit = max(int(100 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = max(int(1000 / chart_data.iloc[-1]['close']), 1)


        learner = ReinforcementLearner(rl_method=args.rl_method, stock_code=stock_code,
                chart_data=chart_data, training_data=training_data,
                min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit,
                delayed_reward_threshold=.05,
                net='actorcritic', num_steps=args.num_steps, lr=0.001,
                value_network_path=None, policy_network_path=policy_network_path,
                output_path=output_path, reuse_models=True)

        learner.run(balance=args.balance)



