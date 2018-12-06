import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Ultra3D-v0',
    entry_point='gym_ultra3d.envs:Ultra3DEnv1D1A',
    #timestep_limit=1000,
    #reward_threshold=1.0,
    #nondeterministic = True,
)

register(
    id='Ultra3D-v1',
    entry_point='gym_ultra3d.envs:Ultra3DEnv2A',
    #timestep_limit=1000,
    #reward_threshold=1.0,
    #nondeterministic = True,
)

register(
    id='Ultra3D-v2',
    entry_point='gym_ultra3d.envs:Ultra3DEnv2A1D',
    #timestep_limit=1000,
    #reward_threshold=1.0,
    #nondeterministic = True,
)