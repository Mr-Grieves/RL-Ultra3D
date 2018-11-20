import numpy as np
import gym
import gym_ultra3d

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten
from keras.optimizers import Adam

import sys
sys.path.append('/home/nathanvw/dev/RL/keras-rl')

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'Ultra3D-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(1)
env.seed(1)
nb_actions = env.action_space.n
print(nb_actions)


# Next, we build a very simple model.
"""
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())"""
print(env.observation_space.shape)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu',input_shape=(1,)+env.observation_space.shape,data_format="channels_first"))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu',data_format="channels_first"))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_first"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_actions,activation='softmax'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=40,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

training = True
testing = False
if training:
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=100, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

if testing:
    dqn.load_weights('dqn_Ultra3D-v0_weights_19-11-18.h5f')

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.fit(env, nb_steps=100, visualize=False, verbose=2)
