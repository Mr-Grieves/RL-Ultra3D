import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Ultra3D-v0',
    entry_point='gym_ultra3d.envs:Ultra3DEnv',
    #timestep_limit=1000,
    #reward_threshold=1.0,
    #nondeterministic = True,
)
