import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, LSTM, Lambda, BatchNormalization, GaussianNoise, Flatten


class ActorNetwork:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr, tau, num_steps):
        self.inp_dim = inp_dim
        self.act_dim = out_dim
        self.tau = tau
        self.lr = lr
        self.num_steps = num_steps
        self.model = self.network()
        self.model2=self.network2()
        self.target_model = self.network()
        self.adam_optimizer = self.optimizer()

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for conti/nuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
         """
        inp = Input((self.num_steps, self.inp_dim,))
        """
        # DNN
        output = Dense(256, activation='sigmoid', 
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(128, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)

        """
        # LSTM
        output = LSTM(256, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = LSTM(128, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(64, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(32, dropout=0.1,
                      stateful=False,
                      kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)

        """
        # ORIGINAL
        # x = Dense(256, activation='relu')(inp)
        # x = GaussianNoise(1.0)(x)
        # #
        # #x = Flatten()(x)
        # x = Dense(128, activation='relu')(x)
        # x = GaussianNoise(1.0)(x)

        """
        output = Dense(self.act_dim, activation='sigmoid', kernel_initializer='random_normal')(output)

        # out = Lambda(lambda i: i)(out)

        return Model(inp, output)

    def network2(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for conti/nuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
         """
        inp = Input(( 1, self.inp_dim,))
        """
        # DNN
        output = Dense(256, activation='sigmoid', 
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(128, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)

        """
        # LSTM
        output = LSTM(256, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = LSTM(128, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(64, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(32, dropout=0.1,
                      stateful=False,
                      kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)

        """
        # ORIGINAL
        # x = Dense(256, activation='relu')(inp)
        # x = GaussianNoise(1.0)(x)
        # #
        # #x = Flatten()(x)
        # x = Dense(128, activation='relu')(x)
        # x = GaussianNoise(1.0)(x)

        """
        output = Dense(self.act_dim, activation='sigmoid', kernel_initializer='random_normal')(output)

        # out = Lambda(lambda i: i)(out)

        return Model(inp, output)

    def predict2(self, sample):
        """ Action prediction
        """
        sample = np.array(sample).reshape(-1,1, self.inp_dim)
        return self.model2.predict(sample)

    def predict(self, sample):
        """ Action prediction
        """
        sample = np.array(sample).reshape(-1, self.num_steps, self.inp_dim)
        return self.model.predict(sample)
    def target_predict(self, sample):
        """ Action prediction (target network)
        """
        sample = np.array(sample).reshape(-1, self.num_steps, self.inp_dim)
        return self.target_model.predict(sample)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def train(self, states, grads):
        """ Actor Training
        """
        states = np.array(states).reshape(-1, self.num_steps, self.inp_dim)
        self.adam_optimizer([states, grads])

    def optimizer(self):
        """ Actor Optimizer
        """

        action_gdts = K.placeholder(shape=(None, 1, self.act_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)][1:])

    def save_model(self, model_path):
        self.model.save_weights(model_path)
        # self.target_model.save_weights(path)

    def load_model(self, model_path):
        self.model.load_weights(model_path)
        # self.target_model.load_weights(path)
