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
parser.add_argument('--verbose', action="store_true", default=False)
parser.add_argument('--noplots', action="store_true", default=False)
args = parser.parse_args()

''' --- For the 1d 1a version --- '''
# ENV_NAME = 'Ultra3D-v0'
# SAVED_WEIGHT_FILE = 'dqn_{}_weights_23-11-18.h5f'.format(ENV_NAME)

''' --- For the 2a version --- '''
#ENV_NAME = 'Ultra3D-v1'
#SAVED_WEIGHT_FILE = 'dqn_{}_weights_02-12-18.h5f'.format(ENV_NAME)
#NB_STEPS = 500000

''' --- For the 2a1d version --- '''
ENV_NAME = 'Ultra3D-v3'
SAVED_WEIGHT_FILE = 'dqn_{}_weights_14-12-18.h5f'.format(ENV_NAME)
NB_STEPS = 500000

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(1)
env.seed(3)
env.set_verbosity(args.verbose)
nb_actions = env.action_space.n

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

memory = SequentialMemory(limit=NB_STEPS, window_length=WINDOW_LENGTH)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.05, nb_steps=NB_STEPS)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=NB_STEPS/100, gamma=.9,
               target_model_update=NB_STEPS/100, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if args.mode == 'train':
    weights_filename = 'weights/dqn_{}_weights.h5f'.format(ENV_NAME)
    checkpoint_weights_filename = 'weights/dqn_' + ENV_NAME + '_weights_{step}.h5f'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=NB_STEPS/10)]

    dqn.fit(env, nb_steps=NB_STEPS, visualize=False, callbacks=callbacks, verbose=1)
    dqn.save_weights(weights_filename, overwrite=True)
    env.print_outcomes()

elif args.mode == 'test':
    weights_filename = 'weights/'+SAVED_WEIGHT_FILE
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10,visualize=not args.noplots, verbose=1)