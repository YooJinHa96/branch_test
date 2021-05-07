import os
import threading
import numpy as np
import argparse


class DummyGraph:
    def as_default(self): return self

    def __enter__(self): pass

    def __exit__(self, type, value, traceback): pass


def set_session(sess): pass


graph = DummyGraph()
sess = None

if os.environ['KERAS_BACKEND'] == 'tensorflow':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten, concatenate
    from tensorflow.keras.initializers import RandomUniform
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers import SGD
#    from tensorflow.keras.backend import set_session
    import keras.backend as K
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    graph = tf.compat.v1.get_default_graph()
    sess = tf.compat.v1.Session()
elif os.environ['KERAS_BACKEND'] == 'plaidml.keras.backend':
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from keras.layers import concatenate
    from keras.optimizers import Adam
    from keras.initializers import RandomUniform
    from keras.optimizers import SGD


class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001,
                 shared_network=None, activation='sigmoid', loss='mse'):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.model = None
        self.model2 = None
        self.target_model1 = None
        self.target_model2 = None
        self.target_model = None

    def predict(self, sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.model.predict(sample).flatten()

    def target_model_predict(self, sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.target_model.predict(sample).flatten()

    def target_model1_predict(self, sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.target_model1.predict(sample).flatten()

    def target_model2_predict(self, sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.target_model2.predict(sample).flatten()

    def train_on_batch(self, x, y):
        loss = 0.
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                loss = self.model.train_on_batch(x, y)
        return loss

    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.output_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0):
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            if net == 'dnn':
                return DNN.get_network_head(Input((input_dim,)))
            elif net == 'lstm':
                return LSTMNetwork.get_network_head(
                    Input((num_steps, input_dim)))



class DNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.input_dim,))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
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
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):

        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)


class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            self.num_steps = num_steps
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.num_steps, self.input_dim))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
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
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (1, self.num_steps, self.input_dim))
        return super().predict(sample)



class ActorNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = 0.001
        self.actor_loss = None
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            self.num_steps = num_steps
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.num_steps, self.input_dim))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.target_model1 = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)
            self.target_model1.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)
            # initialize optimizer
            self.optimizer = Adam(self.lr)

            # copy the weights for initialization
            weights_ = self.model.get_weights()
            self.target_model.set_weights(weights_)

    @staticmethod
    def get_network_head(inp):
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
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def train(self,obs,critic):
        with tf.GradientTape() as tape:
            actions = self.model(obs)
            actor_loss = -tf.reduce_mean(critic([obs, actions]))

        actor_grad = tape.gradient(actor_loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(actor_grad, self.model.trainable_variables))

        self.actor_loss = actor_loss

        return self.actor_loss

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (1, self.num_steps, self.input_dim))
        return super().predict(sample)

    def target_model1_predict(self, sample):
        """ Action prediction (target network)
        """
        sample = np.array(sample).reshape(
            (1, self.num_steps, self.input_dim))
        return super().target_model1_predict(sample)

    def optimizer(self):
        """ Actor Optimizer
        """
        action_gdts = K.placeholder(shape=(None, self.output_dim))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.model.trainable_weights)
        return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.lr).apply_gradients(grads)])

    def target_update(self):
        """ soft target update for training target actor network
        		"""
        weights, weights_t = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(weights)):
            weights_t[i] = self.tau * weights[i] + (1 - self.tau) * weights_t[i]
        self.target_model.set_weights(weights_t)


class CriticNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = 0.001
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            self.num_steps = num_steps
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.num_steps, self.input_dim))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model2 = Model(inp,output)
            self.target_model1 = Model(inp, output)
            self.target_model2 = Model(inp, output)

        self.optimizer1, self.optimizer2 = Adam(self.lr), Adam(self.lr)

        # copy the weights for initialization
        weights_ = self.model.get_weights(), self.model2.get_weights()
        self.target_model1.set_weights(weights_[0]);
        self.target_model2.set_weights(weights_[1])

        self.critic_loss = None

    @staticmethod
    def get_network_head(inp):

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
        return Model(inp, output)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (1, self.num_steps, self.input_dim))
        return super().predict(sample)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def train(self,obs,acts,target):
        """Train Q-network for critic on sampled batch
        		"""
        with tf.GradientTape() as tape1:
            q1_values = self.model([obs, acts], training=True)
            critic_loss_1 = tf.reduce_mean(tf.math.square(q1_values - target))
        critic_grad_1 = tape1.gradient(critic_loss_1, self.model.trainable_variables)  # compute critic gradient
        self.optimizer1.apply_gradients(zip(critic_grad_1, self.model.trainable_variables))

        with tf.GradientTape() as tape2:
            q2_values = self.model2([obs, acts], training=True)
            critic_loss_2 = tf.reduce_mean(tf.math.square(q2_values - target))
        critic_grad_2 = tape2.gradient(critic_loss_2, self.model2.trainable_variables)  # compute critic gradient

        self.optimizer2.apply_gradients(zip(critic_grad_2, self.model2.trainable_variables))

        tf.print("critic loss :", critic_loss_1, critic_loss_2)
        self.critic_loss = float(min(critic_loss_1, critic_loss_2))
        return self.critic_loss
    # tf.print("critic loss :",critic_loss_1)
    # self.critic_loss = float(critic_loss_1)

    def target_model1_predict(self, sample):
        sample = np.array(sample).reshape(
            (1, self.num_steps, self.input_dim))
        return super().target_model1_predict(sample)

    def target_model2_predict(self, sample):
        sample = np.array(sample).reshape(
            (1, self.num_steps, self.input_dim))
        return super().target_model2_predict(sample)

    def target_update(self):
        """  soft target update for training target critic network
        """
        weights, weights_t = self.model.get_weights(), self.target_model1.get_weights()
        for i in range(len(weights)):
            weights_t[i] = self.tau * weights[i] + (1 - self.tau) * weights_t[i]
        self.target_model1.set_weights(weights_t)

        weights, weights_t = self.model2.get_weights(), self.target_model2.get_weights()
        for i in range(len(weights)):
            weights_t[i] = self.tau * weights[i] + (1 - self.tau) * weights_t[i]
        self.target_model2.set_weights(weights_t)