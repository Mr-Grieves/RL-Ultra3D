import numpy as np
import gym
import gym_ultra3d

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten
from keras.optimizers import Adam

import sys
sys.path.append('/home/nathanvw/dev/RL/keras-rl')

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint
import argparse

WINDOW_LENGTH = 4

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train','test'], default='train')
parser.add_argument('--version', choices=['0','1'], default='1')
args = parser.parse_args()

if args.version == 0:
    ENV_NAME = 'Ultra3D-v0'
    weights_filename = 'dqn_{}_weights_23-11-18.h5f'.format(ENV_NAME) if args.mode == 'test' \
                  else 'dqn_{}_weights.h5f'.format(ENV_NAME)
else:
    ENV_NAME = 'Ultra3D-v1'
    weights_filename = 'dqn_{}_weights_28-11-18.h5f'.format(ENV_NAME) if args.mode == 'test' \
                  else 'dqn_{}_weights.h5f'.format(ENV_NAME)

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(10)
env.seed(10)
nb_actions = env.action_space.n
print(nb_actions)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(WINDOW_LENGTH, 128, 128), data_format="channels_first"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', data_format="channels_first"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=50000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, gamma=.9,
               target_model_update=1000, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if args.mode == 'train':
    checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]

    dqn.fit(env, nb_steps=50000, visualize=False, callbacks=callbacks, verbose=1)
    dqn.save_weights(weights_filename, overwrite=True)
    #dqn.test(env, nb_episodes=5,visualize=False, verbose=1)

elif args.mode == 'test':
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10,visualize=True, verbose=1)
