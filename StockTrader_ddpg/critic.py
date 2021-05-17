import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten

class CriticNetwork:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = 1
        self.tau, self.lr = tau, lr
        # Build models and target models
        self.model = self.network()
        self.target_model = self.network()
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))
        #self.action_grads = K.function([self.model.input, self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input((self.env_dim,))
        action = Input((self.act_dim,))
        x = Dense(256, activation='relu')(state)
        x = concatenate([state, action])
        x = Dense(128, activation='relu')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        return Model([state, action], out)

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        action_s = np.array(actions).reshape(-1,1)
        return self.action_grads([states, action_s])

    def target_predict(self, sample, action):
        """ Predict Q-Values using the target network
        """

        sample = np.array(sample).reshape(-1, self.env_dim)

        #print(np.array(sample).reshape(-1,5,self.env_dim).shape)
        action = np.array(action).reshape(-1, 1)
        return self.target_model.predict([sample, action])

    def predict(self, sample, action):
        """ Predict Q-Values using the target network
        """
        sample = np.array(sample).reshape(-1,self.env_dim)
        action = np.array(action).reshape(-1 ,1)
        return self.model.predict([sample, action])

    def train_on_batch(self, samples, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        samples = np.array(samples).reshape(-1,self.env_dim)
        actions = np.array(actions).reshape(-1,1)
        return self.model.train_on_batch([samples, actions], critic_target)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def save_model(self, model_path):
        self.model.save_weights(model_path)
        #self.target_model.save_weights(path)

    def load_model(self, model_path):
        self.model.load_weights(model_path)
       # self.target_model.load_weights(path)
